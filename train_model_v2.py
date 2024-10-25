from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Callable, Any, Dict, Union
import logging
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from model import HybridModel, MultiModalConfig, SequencePredictorConfig, HybridModelConfig

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    window_size: int
    sequence_length: int
    num_mel_bins: int
    height: int
    width: int
    embedding_size: int
    epochs: int
    learning_rate: float
    batch_size: int
    lstm_hidden_size: int
    lstm_num_layers: int
    prediction_weight: float
    num_workers: int
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class PathConfig:
    """Configuration for file paths."""
    mel_pickle_path: Path
    energy_pickle_path: Path
    best_model_path: Path
    final_model_path: Path
    log_dir: Path
    
    def __post_init__(self):
        """Validate and convert string paths to Path objects."""
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, str):
                setattr(self, field, Path(value))
            if field.endswith('_path'):
                self._ensure_parent_exists(getattr(self, field))

    @staticmethod
    def _ensure_parent_exists(path: Path) -> None:
        """Ensure parent directory exists."""
        path.parent.mkdir(parents=True, exist_ok=True)

class TrainingState:
    """Manages training state and metrics."""
    
    def __init__(self, config: TrainingConfig):
        self.epoch: int = 0
        self.best_loss: float = float('inf')
        self.patience_counter: int = 0
        self.early_stop: bool = False
        self.metrics: Dict[str, List[float]] = {
            'train_loss': [], 'recons_loss': [], 'pred_loss': []
        }
        self.writer = SummaryWriter(log_dir='runs/hybrid_model')

    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update training metrics."""
        for key, value in metrics.items():
            self.metrics[key].append(value)
            self.writer.add_scalar(key, value, self.epoch)

    def should_save_checkpoint(self, current_loss: float) -> bool:
        """Check if current model should be saved."""
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.patience_counter = 0
            return True
        self.patience_counter += 1
        return False

class WindowedDataset(Dataset):
    """Dataset class for windowed data processing."""

    def __init__(
        self,
        mel_pickle_path: Path,
        energy_pickle_path: Path,
        window_size: int,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.window_size = window_size
        self.transform = transform
        self.mel_data, self.energy_data = self._load_data(
            mel_pickle_path, energy_pickle_path
        )
        self._validate_data()
        self.num_windows = len(self.mel_data) - self.window_size + 1

    def _load_data(
        self, mel_path: Path, energy_path: Path
    ) -> Tuple[List[Any], List[Any]]:
        """Load data from pickle files."""
        try:
            mel_data = self._load_pickle(mel_path)
            energy_data = self._load_pickle(energy_path)
            return mel_data, energy_data
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    @staticmethod
    def _load_pickle(path: Path) -> Any:
        """Load data from a pickle file."""
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logging.error(f"Error loading {path}: {e}")
            raise

    def _validate_data(self) -> None:
        """Validate loaded data."""
        if not isinstance(self.mel_data, (list, tuple)):
            raise TypeError("Mel data must be a list or tuple")
        if not isinstance(self.energy_data, (list, tuple)):
            raise TypeError("Energy data must be a list or tuple")
        if len(self.mel_data) != len(self.energy_data):
            raise ValueError("Data length mismatch")
        if self.num_windows <= 0:
            raise ValueError("Window size too large for data")

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get windowed data item."""
        if not 0 <= idx < self.num_windows:
            raise IndexError("Index out of range")

        mel_window = self._process_window(
            self.mel_data[idx:idx + self.window_size]
        )
        energy_window = self._process_window(
            self.energy_data[idx:idx + self.window_size]
        )

        if self.transform:
            mel_window = self.transform(mel_window)
            energy_window = self.transform(energy_window)

        return mel_window, energy_window

    @staticmethod
    def _process_window(window: List[Any]) -> torch.Tensor:
        """Process window data into tensor."""
        if isinstance(window[0], (list, tuple)):
            window = [torch.tensor(frame, dtype=torch.float32) for frame in window]
        return torch.stack(window)

class ModelTrainer:
    """Handles model training and validation."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_config: TrainingConfig,
        path_config: PathConfig,
    ):
        self.model = model
        self.config = train_config
        self.paths = path_config
        self.state = TrainingState(train_config)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=1e-5
        )
        self.device = torch.device(train_config.device)
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.paths.log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )

    def train(self, train_loader: DataLoader) -> None:
        """Train the model."""
        try:
            for epoch in range(self.config.epochs):
                self.state.epoch = epoch
                metrics = self._train_epoch(train_loader)
                self.state.update_metrics(metrics)
                
                if self.state.should_save_checkpoint(metrics['train_loss']):
                    self._save_checkpoint('best')
                
                self._log_epoch_metrics(metrics)
                
                if self.state.early_stop:
                    logging.info("Early stopping triggered")
                    break
                    
            self._save_checkpoint('final')
            
        except Exception as e:
            logging.error(f"Training error: {e}")
            raise

    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        metrics = {
            'train_loss': 0.0,
            'recons_loss': 0.0,
            'pred_loss': 0.0
        }
        
        for batch_idx, (mel_windows, energy_windows) in enumerate(train_loader):
            batch_metrics = self._train_step(mel_windows, energy_windows)
            
            for key in metrics:
                metrics[key] += batch_metrics[key]
            
            if batch_idx % 10 == 0:
                self._log_batch_metrics(batch_idx, len(train_loader), batch_metrics)

        return {k: v / len(train_loader) for k, v in metrics.items()}

    def _train_step(
        self, mel_windows: torch.Tensor, energy_windows: torch.Tensor
    ) -> Dict[str, float]:
        """Perform one training step."""
        mel_windows = mel_windows.to(self.device)
        energy_windows = energy_windows.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(mel_windows, energy_windows)
        loss, recons_loss, pred_loss = self._compute_loss(outputs)
        
        loss.backward()
        self.optimizer.step()

        return {
            'train_loss': loss.item(),
            'recons_loss': recons_loss.item(),
            'pred_loss': pred_loss.item()
        }

    def _compute_loss(
        self, outputs: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute model losses."""
        (mel_recon, energy_recon, actual_mel, actual_energy,
         predicted_latent, actual_latent) = outputs

        return hybrid_elbo_loss(
            recon_mel=mel_recon,
            recon_energy=energy_recon,
            mel=actual_mel,
            energy=actual_energy,
            predicted_latent=predicted_latent,
            actual_latent=actual_latent,
            prediction_weight=self.config.prediction_weight
        )

    def _save_checkpoint(self, checkpoint_type: str) -> None:
        """Save model checkpoint."""
        path = (self.paths.best_model_path if checkpoint_type == 'best'
                else self.paths.final_model_path)
        torch.save({
            'epoch': self.state.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.state.best_loss,
            'metrics': self.state.metrics
        }, path)
        logging.info(f"Saved {checkpoint_type} checkpoint to {path}")

    @staticmethod
    def _log_batch_metrics(
        batch_idx: int, num_batches: int, metrics: Dict[str, float]
    ) -> None:
        """Log batch training metrics."""
        logging.info(
            f"Batch [{batch_idx}/{num_batches}] "
            f"Loss: {metrics['train_loss']:.4f} "
            f"Recons Loss: {metrics['recons_loss']:.4f} "
            f"Pred Loss: {metrics['pred_loss']:.4f}"
        )

    @staticmethod
    def _log_epoch_metrics(metrics: Dict[str, float]) -> None:
        """Log epoch training metrics."""
        logging.info(
            f"Epoch Average - "
            f"Loss: {metrics['train_loss']:.4f} "
            f"Recons Loss: {metrics['recons_loss']:.4f} "
            f"Pred Loss: {metrics['pred_loss']:.4f}"
        )

def main() -> None:
    """Main function to run training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Hybrid CVAE-LSTM Model")
    # Add arguments (same as before)
    args = parser.parse_args()

    train_config = TrainingConfig(
        window_size=args.window_size,
        sequence_length=args.sequence_length,
        num_mel_bins=args.num_mel_bins,
        height=args.height,
        width=args.width,
        embedding_size=args.embedding_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_num_layers=args.lstm_num_layers,
        prediction_weight=args.prediction_weight,
        num_workers=args.num_workers
    )

    path_config = PathConfig(
        mel_pickle_path=args.mel_pickle_path,
        energy_pickle_path=args.energy_pickle_path,
        best_model_path=args.best_model_path,
        final_model_path=args.final_model_path,
        log_dir=Path("logs")
    )

    try:
        model = HybridModel(
            window_size=train_config.window_size,
            num_mel_bins=train_config.num_mel_bins,
            height=train_config.height,
            width=train_config.width,
            embedding_size=train_config.embedding_size,
            lstm_hidden_size=train_config.lstm_hidden_size,
            lstm_num_layers=train_config.lstm_num_layers,
            sequence_length=train_config.sequence_length
        ).to(train_config.device)

        dataset = WindowedDataset(
            mel_pickle_path=path_config.mel_pickle_path,
            energy_pickle_path=path_config.energy_pickle_path,
            window_size=train_config.window_size
        )

        data_loader = DataLoader(
            dataset,
            batch_size=train_config.batch_size,
            shuffle=True,
            num_workers=train_config.num_workers
        )

        trainer = ModelTrainer(model, train_config, path_config)
        trainer.train(data_loader)

    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()