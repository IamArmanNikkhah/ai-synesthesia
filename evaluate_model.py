import argparse
import csv
import logging
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import HybridModel  # Import the HybridModel from model.py

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class WindowedDataset(Dataset):
    """
    Dataset class that provides rolling windows of Mel spectrograms and energy maps.

    This class loads Mel spectrogram and energy map data from pickle files,
    generates rolling windows based on a specified window size, and returns
    tuples of (mel_windows, energy_windows) suitable for evaluating the HybridModel.
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


def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    output_csv_path: str,
    prediction_weight: float = 1.0,
):
    """
    Evaluate the HybridModel on test data and save loss metrics to a CSV file.

    Args:
        model (torch.nn.Module): The trained HybridModel.
        data_loader (DataLoader): DataLoader for test data.
        device (torch.device): Device to run the evaluation on.
        output_csv_path (str): Path to save the CSV file with loss metrics.
        prediction_weight (float, optional): Weight for the sequence prediction loss. Defaults to 1.0.
    """
    model.eval()
    metrics = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader, start=1):
            mel_windows, energy_windows = batch
            mel_windows = mel_windows.to(device)
            energy_windows = energy_windows.to(device)

            try:
                (
                    mel_recon,
                    energy_recon,
                    actual_mel,
                    actual_energy,
                    predicted_latent,
                    actual_latent,
                ) = model(mel_windows, energy_windows)
            except Exception as e:
                logging.error(f"Error during model forward pass: {e}")
                continue

            # Compute Reconstruction Losses
            recons_loss_mel = F.mse_loss(mel_recon, actual_mel, reduction='none').mean(dim=[2, 3]).cpu().numpy()
            recons_loss_energy = F.mse_loss(energy_recon, actual_energy, reduction='none').mean(dim=[2, 3, 4]).cpu().numpy()

            # Compute Prediction Loss
            prediction_loss = F.mse_loss(predicted_latent, actual_latent, reduction='none').mean(dim=1).cpu().numpy()

            # Total Loss
            total_loss = recons_loss_mel.mean() + recons_loss_energy.mean() + prediction_weight * prediction_loss.mean()

            for i in range(mel_recon.size(0)):
                metrics.append({
                    'batch_index': batch_idx,
                    'sample_index': i + 1,
                    'mel_reconstruction_loss': recons_loss_mel[i].mean(),
                    'energy_reconstruction_loss': recons_loss_energy[i].mean(),
                    'prediction_loss': prediction_loss[i],
                    'total_loss': total_loss
                })

            logging.info(f'Processed batch {batch_idx}')

    # Write metrics to CSV
    fieldnames = ['batch_index', 'sample_index', 'mel_reconstruction_loss', 'energy_reconstruction_loss', 'prediction_loss', 'total_loss']
    try:
        with open(output_csv_path, mode='w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for metric in metrics:
                writer.writerow(metric)
        logging.info(f'Loss metrics saved to {output_csv_path}')
    except Exception as e:
        logging.error(f'Failed to write CSV file: {e}')
        raise


def main():
    """Main function to parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Hybrid CVAE-LSTM Model and Generate Loss Metrics CSV")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model state_dict")
    parser.add_argument("--window_size", type=int, default=4, help="Window size for input data")
    parser.add_argument("--sequence_length", type=int, default=2, help="Sequence length for LSTM input")
    parser.add_argument("--num_mel_bins", type=int, default=128, help="Number of Mel bins in the spectrogram")
    parser.add_argument("--height", type=int, default=64, help="Height of the energy map")
    parser.add_argument("--width", type=int, default=64, help="Width of the energy map")
    parser.add_argument("--embedding_size", type=int, default=32, help="Size of the latent embedding for each modality")
    parser.add_argument("--lstm_hidden_size", type=int, default=64, help="Hidden size for LSTM")
    parser.add_argument("--lstm_num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--prediction_weight", type=float, default=1.0, help="Weight for the sequence prediction loss")
    parser.add_argument("--output_csv_path", type=str, default="evaluation_metrics.csv", help="Path to save the loss metrics CSV")
    parser.add_argument("--mel_pickle_path", type=str, required=True, help="Path to Mel spectrogram pickle file")
    parser.add_argument("--energy_pickle_path", type=str, required=True, help="Path to energy map pickle file")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Initialize the model
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

    # Load the saved model state_dict
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        logging.info(f"Loaded model state_dict from {args.model_path}")
    except Exception as e:
        logging.error(f"Failed to load model state_dict: {e}")
        raise

    try:
        # Create the dataset and data loader
        dataset = WindowedDataset(
            mel_pickle_path=args.mel_pickle_path,
            energy_pickle_path=args.energy_pickle_path,
            window_size=args.window_size,  # Adjust for sequence length
        )
        data_loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )

        evaluate_model(
            model=model,
            data_loader=data_loader,
            device=device,
            output_csv_path=args.output_csv_path,
            prediction_weight=args.prediction_weight,
        )
    except Exception as e:
        logging.error(f"An error occurred during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()
