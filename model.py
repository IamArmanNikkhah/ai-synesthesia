import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class MelEncoder(nn.Module):
    """Encoder for 2D Mel Spectrogram windows."""

    def __init__(self, window_size: int, num_mel_bins: int, embedding_size: int):
        """
        Initialize the MelEncoder.

        Args:
            window_size (int): Size of the window (number of frames).
            num_mel_bins (int): Number of Mel bins.
            embedding_size (int): Size of the latent embedding.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()

        # Dynamically calculate the size for the fully connected layer
        with torch.no_grad():
            x = torch.zeros(1, 1, window_size, num_mel_bins)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            fc_input_size = x.numel()

        self.fc1 = nn.Linear(fc_input_size, 512)
        self.fc_mean = nn.Linear(512, embedding_size)
        self.fc_log_var = nn.Linear(512, embedding_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the MelEncoder.

        Args:
            x (torch.Tensor): Input Mel spectrogram tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Sampled latent vector, mean, and log variance.
        """
        x = x.unsqueeze(1)  # Shape: (batch_size, 1, window_size, num_mel_bins)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)
        sample = self.reparameterize(z_mean, z_log_var)
        return sample, z_mean, z_log_var

    @staticmethod
    def reparameterize(z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from the latent distribution.

        Args:
            z_mean (torch.Tensor): Mean of the latent distribution.
            z_log_var (torch.Tensor): Log variance of the latent distribution.

        Returns:
            torch.Tensor: Sampled latent vector.
        """
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return eps * std + z_mean


class EnergyMapEncoder(nn.Module):
    """Encoder for 3D Energy Map windows."""

    def __init__(self, window_size: int, height: int, width: int, embedding_size: int):
        """
        Initialize the EnergyMapEncoder.

        Args:
            window_size (int): Size of the window (number of frames).
            height (int): Height of the energy map.
            width (int): Width of the energy map.
            embedding_size (int): Size of the latent embedding.
        """
        super().__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2)
        self.flatten = nn.Flatten()

        # Dynamically calculate the size for the fully connected layer
        with torch.no_grad():
            x = torch.zeros(1, 1, window_size, height, width)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            fc_input_size = x.numel()

        self.fc1 = nn.Linear(fc_input_size, 512)
        self.fc_mean = nn.Linear(512, embedding_size)
        self.fc_log_var = nn.Linear(512, embedding_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the EnergyMapEncoder.

        Args:
            x (torch.Tensor): Input energy map tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Sampled latent vector, mean, and log variance.
        """
        x = x.unsqueeze(1)  # Shape: (batch_size, 1, window_size, height, width)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)
        sample = self.reparameterize(z_mean, z_log_var)
        return sample, z_mean, z_log_var

    @staticmethod
    def reparameterize(z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from the latent distribution.

        Args:
            z_mean (torch.Tensor): Mean of the latent distribution.
            z_log_var (torch.Tensor): Log variance of the latent distribution.

        Returns:
            torch.Tensor: Sampled latent vector.
        """
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return eps * std + z_mean


class HierarchicalDecoder(nn.Module):
    """Hierarchical Decoder with Shared and Modality-Specific Layers."""

    def __init__(self, window_size: int, num_mel_bins: int, height: int, width: int, embedding_size: int):
        """
        Initialize the HierarchicalDecoder.

        Args:
            window_size (int): Size of the window (number of frames).
            num_mel_bins (int): Number of Mel bins.
            height (int): Height of the energy map.
            width (int): Width of the energy map.
            embedding_size (int): Size of the latent embedding.
        """
        super().__init__()
        self.shared_fc = nn.Linear(embedding_size * 2, 512)
        self.shared_fc2 = nn.Linear(512, 1024)

        # Decoder for Mel Spectrogram
        self.mel_fc = nn.Linear(1024, 512)
        self.mel_unflatten = nn.Unflatten(1, (32, window_size // 4, num_mel_bins // 4))
        self.mel_deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.mel_deconv2 = nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2)

        # Decoder for Energy Map
        self.energy_fc = nn.Linear(1024, 512)
        self.energy_unflatten = nn.Unflatten(1, (32, window_size // 4, height // 4, width // 4))
        self.energy_deconv1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.energy_deconv2 = nn.ConvTranspose3d(16, 1, kernel_size=2, stride=2)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the HierarchicalDecoder.

        Args:
            z (torch.Tensor): Combined latent vector.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Reconstructed Mel spectrogram window and energy map window.
        """
        shared = F.relu(self.shared_fc(z))
        shared = F.relu(self.shared_fc2(shared))

        # Decode Mel Spectrogram
        mel = F.relu(self.mel_fc(shared))
        mel = self.mel_unflatten(mel)
        mel = F.relu(self.mel_deconv1(mel))
        mel_recon = self.mel_deconv2(mel)
        mel_recon = mel_recon.squeeze(1)  # Shape: (batch_size, window_size, num_mel_bins)

        # Decode Energy Map
        energy = F.relu(self.energy_fc(shared))
        energy = self.energy_unflatten(energy)
        energy = F.relu(self.energy_deconv1(energy))
        energy_recon = self.energy_deconv2(energy)
        energy_recon = energy_recon.squeeze(1)  # Shape: (batch_size, window_size, height, width)

        return mel_recon, energy_recon


class MultiModalCVAE(nn.Module):
    """Multi-Modal Conditional Variational Autoencoder with Separate Encoders."""

    def __init__(self, window_size: int, num_mel_bins: int, height: int, width: int, embedding_size: int):
        """
        Initialize the MultiModalCVAE.

        Args:
            window_size (int): Size of the window (number of frames).
            num_mel_bins (int): Number of Mel bins.
            height (int): Height of the energy map.
            width (int): Width of the energy map.
            embedding_size (int): Size of the latent embedding.
        """
        super().__init__()
        self.mel_encoder = MelEncoder(window_size, num_mel_bins, embedding_size)
        self.energy_encoder = EnergyMapEncoder(window_size, height, width, embedding_size)
        self.decoder = HierarchicalDecoder(window_size, num_mel_bins, height, width, embedding_size)

    def forward(
        self, mel_window: torch.Tensor, energy_window: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the MultiModalCVAE.

        Args:
            mel_window (torch.Tensor): Input Mel spectrogram window tensor.
            energy_window (torch.Tensor): Input energy map window tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                Reconstructed Mel spectrogram window,
                Reconstructed energy map window,
                Concatenated mean vectors,
                Concatenated log variance vectors.
        """
        # Encode both modalities
        mel_z, mel_mu, mel_log_var = self.mel_encoder(mel_window)
        energy_z, energy_mu, energy_log_var = self.energy_encoder(energy_window)

        # Concatenate latent vectors
        z = torch.cat((mel_z, energy_z), dim=1)
        z_mean = torch.cat((mel_mu, energy_mu), dim=1)
        z_log_var = torch.cat((mel_log_var, energy_log_var), dim=1)

        # Decode to reconstruct both modalities
        mel_recon, energy_recon = self.decoder(z)

        return mel_recon, energy_recon, z_mean, z_log_var


class SequencePredictor(nn.Module):
    """LSTM-based Sequence Predictor for Anomaly Detection."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        """
        Initialize the SequencePredictor.

        Args:
            input_size (int): Number of expected features in the input.
            hidden_size (int): Number of features in the hidden state.
            num_layers (int): Number of recurrent layers.
            output_size (int): Number of features in the output.
        """
        super().__init__()
        self.input_size = input_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SequencePredictor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])  # Take the last output
        return output


class HybridModel(nn.Module):
    """Unified Model for CVAE and LSTM-based Anomaly Detection."""

    def __init__(
        self,
        window_size: int,
        num_mel_bins: int,
        height: int,
        width: int,
        embedding_size: int,
        lstm_hidden_size: int,
        lstm_num_layers: int,
        sequence_length: int,
    ):
        """
        Initialize the HybridModel.

        Args:
            window_size (int): Size of the window (number of frames).
            num_mel_bins (int): Number of Mel bins.
            height (int): Height of the energy map.
            width (int): Width of the energy map.
            embedding_size (int): Size of the latent embedding.
            lstm_hidden_size (int): Number of features in the LSTM hidden state.
            lstm_num_layers (int): Number of recurrent layers in LSTM.
            sequence_length (int): Length of the sequence for LSTM input.
        """
        super().__init__()
        self.cvae = MultiModalCVAE(window_size, num_mel_bins, height, width, embedding_size)
        self.sequence_length = sequence_length
        self.sequence_predictor = SequencePredictor(
            input_size=embedding_size * 2,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            output_size=embedding_size * 2,
        )

    def forward(
        self, mel_windows: torch.Tensor, energy_windows: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the HybridModel.

        Args:
            mel_windows (torch.Tensor): Input Mel spectrogram windows tensor.
            energy_windows (torch.Tensor): Input energy map windows tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                Reconstructed Mel spectrogram windows,
                Reconstructed energy map windows,
                Actual Mel spectrogram windows,
                Actual energy map windows,
                Predicted latent vectors,
                Actual latent vectors.
        """
        batch_size, num_windows, *_ = mel_windows.size()
        latent_vectors = []

        # Encode all windows to get latent vectors
        for i in range(num_windows):
            mel_window = mel_windows[:, i, :, :]
            energy_window = energy_windows[:, i, :, :, :]
            _, _, mu, log_var = self.cvae(mel_window, energy_window)
            latent_vector = mu  # Use mean as the latent vector
            latent_vectors.append(latent_vector.unsqueeze(1))  # Shape: (batch_size, 1, embedding_size * 2)

        # Stack latent vectors into a sequence
        latent_sequence = torch.cat(latent_vectors, dim=1)  # Shape: (batch_size, num_windows, embedding_size * 2)

        # Prepare sequences for LSTM input and target
        seq_length = self.sequence_length
        num_sequences = num_windows - seq_length
        if num_sequences <= 0:
            raise ValueError("Not enough windows for the given sequence length.")

        input_sequences = []
        target_latents = []

        for i in range(num_sequences):
            input_seq = latent_sequence[:, i:i + seq_length, :]
            target_latent = latent_sequence[:, i + seq_length, :]
            input_sequences.append(input_seq.unsqueeze(1))
            target_latents.append(target_latent.unsqueeze(1))

        input_sequences = torch.cat(input_sequences, dim=1)
        target_latents = torch.cat(target_latents, dim=1)

        # Reshape for LSTM input
        batch_size_times_num_sequences = batch_size * num_sequences
        lstm_input = input_sequences.view(-1, seq_length, self.sequence_predictor.input_size)
        target_latents = target_latents.view(-1, self.sequence_predictor.output_size)

        # Sequence prediction using LSTM
        predicted_latent = self.sequence_predictor(lstm_input)  # Shape: (batch_size_times_num_sequences, output_size)

        # Decode predicted latent vectors
        mel_recon_list = []
        energy_recon_list = []

        for i in range(predicted_latent.size(0)):
            z = predicted_latent[i, :].unsqueeze(0)
            mel_recon, energy_recon = self.cvae.decoder(z)
            mel_recon_list.append(mel_recon.unsqueeze(0))
            energy_recon_list.append(energy_recon.unsqueeze(0))

        mel_recon_windows = torch.cat(mel_recon_list, dim=0)
        energy_recon_windows = torch.cat(energy_recon_list, dim=0)

        # Reshape back to (batch_size, num_sequences, window_size, ...)
        mel_recon_windows = mel_recon_windows.view(batch_size, num_sequences, -1, mel_windows.size(-1))
        energy_recon_windows = energy_recon_windows.view(
            batch_size, num_sequences, -1, energy_windows.size(-2), energy_windows.size(-1)
        )

        # Actual windows corresponding to the predicted ones
        actual_mel_windows = mel_windows[:, seq_length:, :, :]
        actual_energy_windows = energy_windows[:, seq_length:, :, :, :]

        return (
            mel_recon_windows,
            energy_recon_windows,
            actual_mel_windows,
            actual_energy_windows,
            predicted_latent,
            target_latents,
        )