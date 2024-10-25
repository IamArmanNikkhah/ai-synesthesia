import argparse
import logging
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import HybridModel  # Import the HybridModel from model.py

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class WindowedDataset(Dataset):
    """
    Dataset class that provides rolling windows of Mel spectrograms and energy maps.

    This class loads Mel spectrogram and energy map data from pickle files,
    generates rolling windows based on a specified window size, and returns
    tuples of (mel_windows, energy_windows) suitable for training the HybridModel.
    """

    def __init__(
        self,
        mel_pickle_path: str,
        energy_pickle_path: str,
        window_size: int,
        transform: callable = None,
    ):
        """
        Initialize the WindowedDataset.

        Args:
            mel_pickle_path (str): Path to the pickle file containing Mel spectrogram data.
            energy_pickle_path (str): Path to the pickle file containing energy map data.
            window_size (int): Size of the rolling window.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.window_size = window_size
        self.transform = transform

        # Load Mel spectrogram data
        try:
            with open(mel_pickle_path, 'rb') as f:
                self.mel_data = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Mel spectrogram file not found: {mel_pickle_path}")
        except Exception as e:
            raise IOError(f"Error loading Mel spectrogram data: {e}")

        # Load energy map data
        try:
            with open(energy_pickle_path, 'rb') as f:
                self.energy_data = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Energy map file not found: {energy_pickle_path}")
        except Exception as e:
            raise IOError(f"Error loading energy map data: {e}")

        # Ensure that data is in the correct format (list or numpy array)
        if not isinstance(self.mel_data, (list, tuple)):
            raise TypeError("Mel spectrogram data must be a list or tuple of frames")
        if not isinstance(self.energy_data, (list, tuple)):
            raise TypeError("Energy map data must be a list or tuple of frames")

        # Check that data lengths match
        if len(self.mel_data) != len(self.energy_data):
            raise ValueError("Mel data and energy map data must have the same length")

        self.num_samples = len(self.mel_data)

        # Compute the number of windows
        self.num_windows = self.num_samples - self.window_size + 1
        if self.num_windows <= 0:
            raise ValueError(
                "Window size is larger than the number of samples in the data"
            )

    def __len__(self) -> int:
        """Return the total number of windows."""
        return self.num_windows

    def __getitem__(self, idx: int) -> tuple:
        """
        Get the rolling window at the specified index.

        Args:
            idx (int): Index of the window.

        Returns:
            tuple: Tuple of (mel_window, energy_window)
        """
        if idx < 0 or idx >= self.num_windows:
            raise IndexError("Index out of range")

        # Get the windowed data
        mel_window = self.mel_data[idx: idx + self.window_size]
        energy_window = self.energy_data[idx: idx + self.window_size]

        # Convert lists to numpy arrays if necessary
        if isinstance(mel_window, list):
            mel_window = [frame.astype('float32') for frame in mel_window]
        if isinstance(energy_window, list):
            energy_window = [frame.astype('float32') for frame in energy_window]

        # Stack frames along a new dimension to create the window
        mel_window = torch.tensor(mel_window, dtype=torch.float32)  # Shape: (window_size, num_mel_bins)
        energy_window = torch.tensor(
            energy_window, dtype=torch.float32
        )  # Shape: (window_size, height, width)

        # Apply any transforms
        if self.transform:
            mel_window = self.transform(mel_window)
            energy_window = self.transform(energy_window)

        return mel_window, energy_window


def hybrid_elbo_loss(
    recon_mel: torch.Tensor,
    recon_energy: torch.Tensor,
    mel: torch.Tensor,
    energy: torch.Tensor,
    predicted_latent: torch.Tensor,
    actual_latent: torch.Tensor,
    prediction_weight: float = 1.0,
) -> tuple:
    """
    Compute the combined ELBO loss for the HybridModel.

    Args:
        recon_mel (torch.Tensor): Reconstructed Mel spectrogram windows.
        recon_energy (torch.Tensor): Reconstructed energy map windows.
        mel (torch.Tensor): Original Mel spectrogram windows.
        energy (torch.Tensor): Original energy map windows.
        predicted_latent (torch.Tensor): Latent vectors predicted by LSTM.
        actual_latent (torch.Tensor): Actual latent vectors from CVAE.
        prediction_weight (float, optional): Weight for the sequence prediction loss. Defaults to 1.0.

    Returns:
        tuple:
            Total loss,
            Reconstruction loss,
            Sequence prediction loss.
    """
    # Reconstruction losses
    recons_loss_mel = F.mse_loss(recon_mel, mel, reduction='mean')
    recons_loss_energy = F.mse_loss(recon_energy, energy, reduction='mean')
    recons_loss = recons_loss_mel + recons_loss_energy

    # Sequence prediction loss
    prediction_loss = F.mse_loss(predicted_latent, actual_latent, reduction='mean')

    # Total loss
    loss = recons_loss + prediction_weight * prediction_loss
    return loss, recons_loss.detach(), prediction_loss.detach()


def train_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    prediction_weight: float = 1.0,
    best_model_path: str = 'best_model.pth',
    final_model_path: str = 'final_model.pth',
):
    """
    Train the HybridModel.

    Args:
        model (torch.nn.Module): The HybridModel to train.
        data_loader (DataLoader): DataLoader for training data.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        device (torch.device): Device to train on.
        prediction_weight (float, optional): Weight for the sequence prediction loss. Defaults to 1.0.
        best_model_path (str, optional): Path to save the best model. Defaults to 'best_model.pth'.
        final_model_path (str, optional): Path to save the final model. Defaults to 'final_model.pth'.
    """
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-5)
    model.train()
    best_loss = float('inf')

    for epoch in range(num_epochs):
        train_loss_avg = 0.0
        recons_loss_avg = 0.0
        pred_loss_avg = 0.0
        num_batches = 0

        for batch in data_loader:
            mel_windows, energy_windows = batch
            mel_windows = mel_windows.to(device)
            energy_windows = energy_windows.to(device)

            (
                mel_recon,
                energy_recon,
                actual_mel,
                actual_energy,
                predicted_latent,
                actual_latent,
            ) = model(mel_windows, energy_windows)

            loss, recons_loss, prediction_loss = hybrid_elbo_loss(
                recon_mel=mel_recon,
                recon_energy=energy_recon,
                mel=actual_mel,
                energy=actual_energy,
                predicted_latent=predicted_latent,
                actual_latent=actual_latent,
                prediction_weight=prediction_weight,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_avg += loss.item()
            recons_loss_avg += recons_loss.item()
            pred_loss_avg += prediction_loss.item()
            num_batches += 1

            if num_batches % 10 == 0:
                logging.info(
                    f'Epoch [{epoch + 1}/{num_epochs}] Batch [{num_batches}] Loss: {loss.item():.4f} '
                    f'Recons Loss: {recons_loss.item():.4f} Prediction Loss: {prediction_loss.item():.4f}'
                )

        train_loss_avg /= num_batches
        recons_loss_avg /= num_batches
        pred_loss_avg /= num_batches
        logging.info(
            f'Epoch [{epoch + 1}/{num_epochs}] Average Loss: {train_loss_avg:.4f} '
            f'Avg Recons Loss: {recons_loss_avg:.4f} Avg Prediction Loss: {pred_loss_avg:.4f}'
        )

        # Save the model if it has the lowest loss so far
        if train_loss_avg < best_loss:
            best_loss = train_loss_avg
            torch.save(model.state_dict(), best_model_path)
            logging.info(f'New best model saved at epoch {epoch + 1} with loss {best_loss:.4f}')

    # Save the final model after training
    torch.save(model.state_dict(), final_model_path)
    logging.info(f'Final model saved after epoch {num_epochs}')


def main():
    """Main function to parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Hybrid CVAE-LSTM Model with Rolling Window Approach")
    parser.add_argument("--window_size", type=int, default=4, help="Window size for input data")
    parser.add_argument("--sequence_length", type=int, default=2, help="Sequence length for LSTM input")
    parser.add_argument("--num_mel_bins", type=int, default=128, help="Number of Mel bins in the spectrogram")
    parser.add_argument("--height", type=int, default=64, help="Height of the energy map")
    parser.add_argument("--width", type=int, default=64, help="Width of the energy map")
    parser.add_argument(
        "--embedding_size", type=int, default=32, help="Size of the latent embedding for each modality"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lstm_hidden_size", type=int, default=64, help="Hidden size for LSTM")
    parser.add_argument("--lstm_num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument(
        "--prediction_weight", type=float, default=1.0, help="Weight for the sequence prediction loss"
    )
    parser.add_argument("--best_model_path", type=str, default="best_model.pth", help="Path to save the best model")
    parser.add_argument("--final_model_path", type=str, default="final_model.pth", help="Path to save the final model")
    parser.add_argument("--mel_pickle_path", type=str, required=True, help="Path to Mel spectrogram pickle file")
    parser.add_argument("--energy_pickle_path", type=str, required=True, help="Path to energy map pickle file")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = HybridModel(
        window_size=args.window_size,
        num_mel_bins=args.num_mel_bins,
        height=args.height,
        width=args.width,
        embedding_size=args.embedding_size,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_num_layers=args.lstm_num_layers,
        sequence_length=args.sequence_length,
    ).to(device)

    try:
        # Create the dataset and data loader
        dataset = WindowedDataset(
            mel_pickle_path=args.mel_pickle_path,
            energy_pickle_path=args.energy_pickle_path,
            window_size=args.window_size,  # Adjust for sequence length
        )
        data_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
        )

        train_model(
            model=model,
            data_loader=data_loader,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=device,
            prediction_weight=args.prediction_weight,
            best_model_path=args.best_model_path,
            final_model_path=args.final_model_path,
        )
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
