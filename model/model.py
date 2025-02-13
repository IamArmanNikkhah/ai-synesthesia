# model/model.py
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EncoderConfig:
    window_size: int
    embedding_size: int
    hidden_size: int = 512
    conv1_channels: int = 16
    conv2_channels: int = 32
    kernel_size: int = 3
    padding: int = 1

@dataclass
class EnergyMapConfig:
    window_size: int
    height: int
    width: int
    embedding_size: int
    hidden_size: int = 512
    conv1_channels: int = 16
    conv2_channels: int = 32
    kernel_size: int = 3
    padding: int = 1

@dataclass
class DecoderConfig:
    window_size: int
    num_mel_bins: int
    height: int
    width: int
    embedding_size: int
    hidden_size: int = 512
    shared_hidden_size: int = 1024
    conv_channels: int = 32
    mel_intermediate: tuple = (4, 4)
    energy_intermediate: tuple = (4, 4, 4)

@dataclass
class MultiModalConfig:
    window_size: int
    num_mel_bins: int
    height: int
    width: int
    embedding_size: int
    hidden_size: int = 512

@dataclass
class SequencePredictorConfig:
    input_size: int
    hidden_size: int
    num_layers: int
    output_size: int
    dropout_rate: float = 0.1

@dataclass
class HybridModelConfig:
    multimodal_config: MultiModalConfig
    sequence_config: SequencePredictorConfig
    sequence_length: int

# ---------------------------------------------------------------------------
# MelEncoder using AdaptiveAvgPool2d
# ---------------------------------------------------------------------------

class MelEncoder(nn.Module):
    def __init__(self, config: EncoderConfig, num_mel_bins: int):
        super().__init__()
        if config.window_size <= 0 or config.embedding_size <= 0:
            raise ValueError("Window size and embedding size must be positive integers")
        self.config = config
        self.num_mel_bins = num_mel_bins
        self.conv1 = nn.Conv2d(1, config.conv1_channels, kernel_size=config.kernel_size, padding=config.padding)
        self.conv2 = nn.Conv2d(config.conv1_channels, config.conv2_channels, kernel_size=config.kernel_size, padding=config.padding)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(config.conv2_channels, config.hidden_size)
        self.fc_mean = nn.Linear(config.hidden_size, config.embedding_size)
        self.fc_log_var = nn.Linear(config.hidden_size, config.embedding_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError("Input tensor must have 3 dimensions (batch, window_size, num_mel_bins)")
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, start_dim=1)
        hidden = F.relu(self.fc1(x))
        z_mean = self.fc_mean(hidden)
        z_log_var = self.fc_log_var(hidden)
        z = self.reparameterize(z_mean, z_log_var)
        return z, z_mean, z_log_var

    def reparameterize(self, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return eps * std + z_mean

# ---------------------------------------------------------------------------
# EnergyMapEncoder using AdaptiveAvgPool3d
# ---------------------------------------------------------------------------

class EnergyMapEncoder(nn.Module):
    def __init__(self, config: EnergyMapConfig):
        super().__init__()
        if config.window_size <= 0 or config.embedding_size <= 0:
            raise ValueError("Window size and embedding size must be positive integers")
        self.config = config
        self.conv1 = nn.Conv3d(1, config.conv1_channels, kernel_size=config.kernel_size, padding=config.padding)
        self.conv2 = nn.Conv3d(config.conv1_channels, config.conv2_channels, kernel_size=config.kernel_size, padding=config.padding)
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(config.conv2_channels, config.hidden_size)
        self.fc_mean = nn.Linear(config.hidden_size, config.embedding_size)
        self.fc_log_var = nn.Linear(config.hidden_size, config.embedding_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() != 4:
            raise ValueError("Input tensor must have 4 dimensions (batch, window_size, height, width)")
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, start_dim=1)
        hidden = F.relu(self.fc1(x))
        z_mean = self.fc_mean(hidden)
        z_log_var = self.fc_log_var(hidden)
        z = self.reparameterize(z_mean, z_log_var)
        return z, z_mean, z_log_var

    def reparameterize(self, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return eps * std + z_mean

# ---------------------------------------------------------------------------
# Hierarchical Decoder
# ---------------------------------------------------------------------------

class HierarchicalDecoder(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        self.shared_fc = nn.Linear(config.embedding_size * 2, config.hidden_size)
        self.shared_fc2 = nn.Linear(config.hidden_size, config.shared_hidden_size)
        # Mel branch
        interm_h, interm_w = config.mel_intermediate
        self.mel_intermediate_numel = config.conv_channels * interm_h * interm_w
        self.mel_fc = nn.Linear(config.shared_hidden_size, self.mel_intermediate_numel)
        self.mel_deconv1 = nn.ConvTranspose2d(config.conv_channels, config.conv_channels // 2, kernel_size=2, stride=2)
        self.mel_deconv2 = nn.ConvTranspose2d(config.conv_channels // 2, 1, kernel_size=2, stride=2)
        # Energy branch
        interm_d, interm_h_e, interm_w_e = config.energy_intermediate
        self.energy_intermediate_numel = config.conv_channels * interm_d * interm_h_e * interm_w_e
        self.energy_fc = nn.Linear(config.shared_hidden_size, self.energy_intermediate_numel)
        self.energy_deconv1 = nn.ConvTranspose3d(config.conv_channels, config.conv_channels // 2, kernel_size=2, stride=2)
        self.energy_deconv2 = nn.ConvTranspose3d(config.conv_channels // 2, 1, kernel_size=2, stride=2)

    def _decode_mel(self, shared: torch.Tensor) -> torch.Tensor:
        mel = F.relu(self.mel_fc(shared))
        batch_size = mel.size(0)
        interm_h, interm_w = self.config.mel_intermediate
        mel = mel.view(batch_size, self.config.conv_channels, interm_h, interm_w)
        mel = F.relu(self.mel_deconv1(mel))
        mel = self.mel_deconv2(mel)
        target_size = (self.config.window_size, self.config.num_mel_bins)
        mel = F.interpolate(mel, size=target_size, mode='bilinear', align_corners=False)
        if mel.shape[1] == 1:
            mel = mel.squeeze(1)
        return mel

    def _decode_energy(self, shared: torch.Tensor) -> torch.Tensor:
        energy = F.relu(self.energy_fc(shared))
        batch_size = energy.size(0)
        interm_d, interm_h, interm_w = self.config.energy_intermediate
        energy = energy.view(batch_size, self.config.conv_channels, interm_d, interm_h, interm_w)
        energy = F.relu(self.energy_deconv1(energy))
        energy = self.energy_deconv2(energy)
        target_size = (self.config.window_size, self.config.height, self.config.width)
        energy = F.interpolate(energy, size=target_size, mode='trilinear', align_corners=False)
        if energy.shape[1] == 1:
            energy = energy.squeeze(1)
        return energy

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if z.dim() != 2:
            raise ValueError("Input tensor must have 2 dimensions")
        shared = F.relu(self.shared_fc(z))
        shared = F.relu(self.shared_fc2(shared))
        mel_recon = self._decode_mel(shared)
        energy_recon = self._decode_energy(shared)
        return mel_recon, energy_recon

# ---------------------------------------------------------------------------
# MultiModal CVAE
# ---------------------------------------------------------------------------

class MultiModalCVAE(nn.Module):
    """Multi-Modal CVAE with separate encoders."""
    def __init__(self, config: MultiModalConfig):
        super().__init__()
        self._validate_config(config)
        self.config = config
        self.mel_encoder = MelEncoder(
            EncoderConfig(window_size=config.window_size, embedding_size=config.embedding_size, hidden_size=config.hidden_size),
            num_mel_bins=config.num_mel_bins
        )
        self.energy_encoder = EnergyMapEncoder(
            EnergyMapConfig(window_size=config.window_size, height=config.height, width=config.width,
                            embedding_size=config.embedding_size, hidden_size=config.hidden_size)
        )
        self.decoder = HierarchicalDecoder(
            DecoderConfig(window_size=config.window_size, num_mel_bins=config.num_mel_bins,
                          height=config.height, width=config.width, embedding_size=config.embedding_size,
                          hidden_size=config.hidden_size)
        )

    @staticmethod
    def _validate_config(config: MultiModalConfig) -> None:
        if any(param <= 0 for param in [config.window_size, config.num_mel_bins, config.height, config.width, config.embedding_size]):
            raise ValueError("All dimensions must be positive integers")

    def encode(self, mel_window: torch.Tensor, energy_window: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mel_z, mel_mu, mel_log_var = self.mel_encoder(mel_window)
        energy_z, energy_mu, energy_log_var = self.energy_encoder(energy_window)
        z = torch.cat((mel_z, energy_z), dim=1)
        z_mean = torch.cat((mel_mu, energy_mu), dim=1)
        z_log_var = torch.cat((mel_log_var, energy_log_var), dim=1)
        return z, z_mean, z_log_var

    def forward(self, mel_window: torch.Tensor, energy_window: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_input_shapes(mel_window, energy_window)
        z, z_mean, z_log_var = self.encode(mel_window, energy_window)
        mel_recon, energy_recon = self.decoder(z)
        return mel_recon, energy_recon, z_mean, z_log_var

    def _validate_input_shapes(self, mel_window: torch.Tensor, energy_window: torch.Tensor) -> None:
        if mel_window.dim() != 3:
            raise ValueError("Mel window must have 3 dimensions")
        if energy_window.dim() != 4:
            raise ValueError("Energy window must have 4 dimensions")

# ---------------------------------------------------------------------------
# Sequence Predictor
# ---------------------------------------------------------------------------

class SequencePredictor(nn.Module):
    def __init__(self, config: SequencePredictorConfig):
        super().__init__()
        self._validate_config(config)
        self.config = config
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout_rate if config.num_layers > 1 else 0
        )
        self.fc = nn.Linear(config.hidden_size, config.output_size)

    @staticmethod
    def _validate_config(config: SequencePredictorConfig) -> None:
        if config.num_layers <= 0:
            raise ValueError("Number of layers must be positive")
        if not 0 <= config.dropout_rate <= 1:
            raise ValueError("Dropout rate must be between 0 and 1")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.dim() != 3:
            raise ValueError("Input must have 3 dimensions (batch, sequence, features)")
        if x.size(2) != self.config.input_size:
            raise ValueError(f"Expected input size {self.config.input_size}, got {x.size(2)}")

# ---------------------------------------------------------------------------
# Hybrid Model
# ---------------------------------------------------------------------------

class HybridModel(nn.Module):
    """Unified CVAE + LSTM model for anomaly detection."""
    def __init__(self, config: HybridModelConfig):
        super().__init__()
        self._validate_config(config)
        self.config = config
        self.cvae = MultiModalCVAE(config.multimodal_config)
        self.sequence_predictor = SequencePredictor(config.sequence_config)
        self.sequence_length = config.sequence_length
        # Buffers for latent parameters (for stability monitoring)
        self.last_latent_log_vars = None  # (batch, latent_dim)
        self.last_z_mean = None           # (batch, latent_dim)
        self.last_lstm_input = None       # lstm input used for prediction

    @staticmethod
    def _validate_config(config: HybridModelConfig) -> None:
        if config.sequence_length <= 0:
            raise ValueError("Sequence length must be positive")

    def _prepare_sequences(self, latent_sequence: torch.Tensor, num_windows: int) -> Tuple[torch.Tensor, torch.Tensor]:
        num_sequences = num_windows - self.sequence_length
        if num_sequences <= 0:
            raise ValueError("Not enough windows for the given sequence length")
        input_sequences = []
        target_latents = []
        for i in range(num_sequences):
            input_seq = latent_sequence[:, i:i + self.sequence_length, :]
            target_latent = latent_sequence[:, i + self.sequence_length, :]
            input_sequences.append(input_seq.unsqueeze(1))
            target_latents.append(target_latent.unsqueeze(1))
        return torch.cat(input_sequences, dim=1), torch.cat(target_latents, dim=1)

    def _process_windows(self, mel_windows: torch.Tensor, energy_windows: torch.Tensor) -> torch.Tensor:
        batch_size, num_windows = mel_windows.shape[:2]
        latent_vectors = []
        latent_log_vars = []
        latent_means = []
        for i in range(num_windows):
            mel_window = mel_windows[:, i, :, :]
            energy_window = energy_windows[:, i, :, :, :]
            # Use the CVAE encoder (only the latent mean is used for sequence prediction)
            _, mu, log_var = self.cvae(mel_window, energy_window)
            latent_vectors.append(mu.unsqueeze(1))
            latent_log_vars.append(log_var.unsqueeze(1))
            latent_means.append(mu.unsqueeze(1))
        # Buffer the last window’s latent stats (detached to avoid graph retention)
        self.last_latent_log_vars = torch.cat(latent_log_vars, dim=1)[:, -1, :].detach()
        self.last_z_mean = torch.cat(latent_means, dim=1)[:, -1, :].detach()
        return torch.cat(latent_vectors, dim=1)

    def forward(self, mel_windows: torch.Tensor, energy_windows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._validate_input_shapes(mel_windows, energy_windows)
        batch_size, num_windows = mel_windows.shape[:2]
        latent_sequence = self._process_windows(mel_windows, energy_windows)
        input_sequences, target_latents = self._prepare_sequences(latent_sequence, num_windows)
        lstm_input = self._reshape_for_lstm(input_sequences)
        self.last_lstm_input = lstm_input.detach()
        predicted_latent = self.sequence_predictor(lstm_input)
        mel_recon, energy_recon = self._generate_reconstructions(predicted_latent, batch_size)
        actual_mel = mel_windows[:, self.sequence_length:, :, :]
        actual_energy = energy_windows[:, self.sequence_length:, :, :, :]
        return (
            mel_recon,
            energy_recon,
            actual_mel,
            actual_energy,
            predicted_latent,
            target_latents.view(-1, self.config.multimodal_config.embedding_size * 2)
        )

    def _validate_input_shapes(self, mel_windows: torch.Tensor, energy_windows: torch.Tensor) -> None:
        if mel_windows.dim() != 4:
            raise ValueError("Mel windows must have 4 dimensions")
        if energy_windows.dim() != 5:
            raise ValueError("Energy windows must have 5 dimensions")

    def _reshape_for_lstm(self, input_sequences: torch.Tensor) -> torch.Tensor:
        return input_sequences.view(-1, self.sequence_length, self.config.sequence_config.input_size)

    def _generate_reconstructions(self, predicted_latent: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        mel_recon_list = []
        energy_recon_list = []
        for i in range(predicted_latent.size(0)):
            z = predicted_latent[i, :].unsqueeze(0)
            mel_recon, energy_recon = self.cvae.decoder(z)
            mel_recon_list.append(mel_recon.unsqueeze(0))
            energy_recon_list.append(energy_recon.unsqueeze(0))
        return self._reshape_reconstructions(
            torch.cat(mel_recon_list, dim=0),
            torch.cat(energy_recon_list, dim=0),
            batch_size
        )

    def _reshape_reconstructions(self, mel_recon: torch.Tensor, energy_recon: torch.Tensor, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        num_sequences = mel_recon.size(0) // batch_size
        return (
            mel_recon.view(batch_size, num_sequences, -1, mel_recon.size(-1)),
            energy_recon.view(batch_size, num_sequences, -1, energy_recon.size(-2), energy_recon.size(-1))
        )

    def reset_latent_buffers(self) -> None:
        """Reset latent parameter buffers at the beginning of an epoch."""
        self.last_latent_log_vars = None
        self.last_z_mean = None
        self.last_lstm_input = None
