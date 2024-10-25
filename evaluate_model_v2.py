from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import pickle
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import HybridModel
from train_model import WindowedDataset  # Reuse dataset class from training

@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    model_path: Path
    window_size: int
    sequence_length: int
    num_mel_bins: int
    height: int
    width: int
    embedding_size: int
    lstm_hidden_size: int
    lstm_num_layers: int
    batch_size: int
    prediction_weight: float
    num_workers: int
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class PathConfig:
    """Configuration for file paths."""
    mel_pickle_path: Path
    energy_pickle_path: Path
    output_dir: Path
    
    def __post_init__(self):
        """Create output directory and convert paths."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, str):
                setattr(self, field, Path(value))

class MetricsManager:
    """Manages evaluation metrics computation and storage."""
    
    def __init__(self, output_dir: Path):
        self.metrics: List[Dict[str, float]] = []
        self.output_dir = output_dir
        self.summary_metrics: Dict[str, float] = {}

    def add_batch_metrics(
        self,
        batch_idx: int,
        mel_loss: torch.Tensor,
        energy_loss: torch.Tensor,
        prediction_loss: torch.Tensor,
        total_loss: float
    ) -> None:
        """Add metrics for a batch."""
        mel_loss = mel_loss.cpu().numpy()
        energy_loss = energy_loss.cpu().numpy()
        prediction_loss = prediction_loss.cpu().numpy()

        for i in range(len(mel_loss)):
            self.metrics.append({
                'batch_index': batch_idx,
                'sample_index': i + 1,
                'mel_reconstruction_loss': float(mel_loss[i].mean()),
                'energy_reconstruction_loss': float(energy_loss[i].mean()),
                'prediction_loss': float(prediction_loss[i]),
                'total_loss': float(total_loss)
            })

    def compute_summary_metrics(self) -> None:
        """Compute summary statistics for all metrics."""
        df = pd.DataFrame(self.metrics)
        for column in df.select_dtypes(include=[np.number]).columns:
            self.summary_metrics.update({
                f'{column}_mean': float(df[column].mean()),
                f'{column}_std': float(df[column].std()),
                f'{column}_median': float(df[column].median()),
                f'{column}_min': float(df[column].min()),
                f'{column}_max': float(df[column].max())
            })

    def save_metrics(self) -> None:
        """Save metrics to CSV and summary to JSON."""
        metrics_df = pd.DataFrame(self.metrics)
        metrics_df.to_csv(self.output_dir / 'detailed_metrics.csv', index=False)
        
        summary_df = pd.DataFrame([self.summary_metrics])
        summary_df.to_csv(self.output_dir / 'summary_metrics.csv', index=False)

    def plot_metrics(self) -> None:
        """Generate plots for metrics visualization."""
        df = pd.DataFrame(self.metrics)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        metrics_to_plot = [
            'mel_reconstruction_loss',
            'energy_reconstruction_loss',
            'prediction_loss',
            'total_loss'
        ]
        
        for ax, metric in zip(axes.flat, metrics_to_plot):
            df[metric].hist(ax=ax, bins=50)
            ax.set_title(f'Distribution of {metric}')
            ax.set_xlabel('Loss Value')
            ax.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'metrics_distribution.png')
        plt.close()

class ModelEvaluator:
    """Handles model evaluation process."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: EvaluationConfig,
        paths: PathConfig
    ):
        self.model = model
        self.config = config
        self.paths = paths
        self.device = torch.device(config.device)
        self.metrics_manager = MetricsManager(paths.output_dir)
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.paths.output_dir / 'evaluation.log'),
                logging.StreamHandler()
            ]
        )

    def evaluate(self, data_loader: DataLoader) -> None:
        """Run evaluation process."""
        self.model.eval()
        
        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(data_loader), 1):
                    self._evaluate_batch(batch_idx, batch)

            self.metrics_manager.compute_summary_metrics()
            self._save_results()
            
        except Exception as e:
            logging.error(f"Evaluation failed: {e}")
            raise

    def _evaluate_batch(
        self,
        batch_idx: int,
        batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Evaluate a single batch."""
        mel_windows, energy_windows = batch
        mel_windows = mel_windows.to(self.device)
        energy_windows = energy_windows.to(self.device)

        try:
            outputs = self.model(mel_windows, energy_windows)
            losses = self._compute_losses(outputs)
            self.metrics_manager.add_batch_metrics(batch_idx, *losses)
        except Exception as e:
            logging.error(f"Error processing batch {batch_idx}: {e}")

    def _compute_losses(
        self,
        outputs: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Compute various loss metrics."""
        (mel_recon, energy_recon, actual_mel, actual_energy,
         predicted_latent, actual_latent) = outputs

        mel_loss = F.mse_loss(mel_recon, actual_mel, reduction='none').mean(dim=[2, 3])
        energy_loss = F.mse_loss(energy_recon, actual_energy, reduction='none').mean(dim=[2, 3, 4])
        pred_loss = F.mse_loss(predicted_latent, actual_latent, reduction='none').mean(dim=1)
        
        total_loss = (mel_loss.mean() + energy_loss.mean() + 
                     self.config.prediction_weight * pred_loss.mean())

        return mel_loss, energy_loss, pred_loss, float(total_loss)

    def _save_results(self) -> None:
        """Save evaluation results."""
        self.metrics_manager.save_metrics()
        self.metrics_manager.plot_metrics()
        logging.info(f"Results saved to {self.paths.output_dir}")

def main() -> None:
    """Main function to run evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Hybrid CVAE-LSTM Model")
    # Add arguments (same as before)
    args = parser.parse_args()

    config = EvaluationConfig(
        model_path=Path(args.model_path),
        window_size=args.window_size,
        sequence_length=args.sequence_length,
        num_mel_bins=args.num_mel_bins,
        height=args.height,
        width=args.width,
        embedding_size=args.embedding_size,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_num_layers=args.lstm_num_layers,
        batch_size=args.batch_size,
        prediction_weight=args.prediction_weight,
        num_workers=args.num_workers
    )

    paths = PathConfig(
        mel_pickle_path=args.mel_pickle_path,
        energy_pickle_path=args.energy_pickle_path,
        output_dir=Path("evaluation_results")
    )

    try:
        model = HybridModel(
            window_size=config.window_size,
            num_mel_bins=config.num_mel_bins,
            height=config.height,
            width=config.width,
            embedding_size=config.embedding_size,
            lstm_hidden_size=config.lstm_hidden_size,
            lstm_num_layers=config.lstm_num_layers,
            sequence_length=config.sequence_length
        ).to(config.device)

        model.load_state_dict(torch.load(config.model_path, 
                                       map_location=config.device))

        dataset = WindowedDataset(
            mel_pickle_path=paths.mel_pickle_path,
            energy_pickle_path=paths.energy_pickle_path,
            window_size=config.window_size
        )

        data_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )

        evaluator = ModelEvaluator(model, config, paths)
        evaluator.evaluate(data_loader)

    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()