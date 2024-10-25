from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union, List
from dataclasses import dataclass

@dataclass
class EncoderConfig:
    """Configuration for encoder modules."""
    window_size: int
    embedding_size: int
    hidden_size: int = 512
    conv1_channels: int = 16
    conv2_channels: int = 32
    kernel_size: int = 3
    padding: int = 1
    pool_size: int = 2


@dataclass
class EnergyMapConfig:
    """Configuration for energy map encoder."""
    window_size: int
    height: int
    width: int
    embedding_size: int
    hidden_size: int = 512
    conv1_channels: int = 16
    conv2_channels: int = 32
    kernel_size: int = 3
    padding: int = 1
    pool_size: int = 2

@dataclass
class DecoderConfig:
    """Configuration for hierarchical decoder."""
    window_size: int
    num_mel_bins: int
    height: int
    width: int
    embedding_size: int
    hidden_size: int = 512
    shared_hidden_size: int = 1024
    conv_channels: int = 32



@dataclass
class MultiModalConfig:
    """Configuration for Multi-Modal CVAE."""
    window_size: int
    num_mel_bins: int
    height: int
    width: int
    embedding_size: int
    hidden_size: int = 512

@dataclass
class SequencePredictorConfig:
    """Configuration for Sequence Predictor."""
    input_size: int
    hidden_size: int
    num_layers: int
    output_size: int
    dropout_rate: float = 0.1

@dataclass
class HybridModelConfig:
    """Configuration for Hybrid Model."""
    multimodal_config: MultiModalConfig
    sequence_config: SequencePredictorConfig
    sequence_length: int





class BaseEncoder(nn.Module, ABC):
    """Abstract base class for encoders."""
    
    @abstractmethod
    def reparameterize(self, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick implementation."""
        pass

    @staticmethod
    def validate_inputs(window_size: int, embedding_size: int) -> None:
        """Validate input parameters."""
        if window_size <= 0 or embedding_size <= 0:
            raise ValueError("Window size and embedding size must be positive integers")

class MelEncoder(BaseEncoder):
    """Encoder for 2D Mel Spectrogram windows."""

    def __init__(self, config: EncoderConfig, num_mel_bins: int):
        """Initialize the MelEncoder."""
        super().__init__()
        self.validate_inputs(config.window_size, config.embedding_size)
        
        self.conv1 = nn.Conv2d(1, config.conv1_channels, 
                              kernel_size=config.kernel_size, 
                              padding=config.padding)
        self.conv2 = nn.Conv2d(config.conv1_channels, 
                              config.conv2_channels, 
                              kernel_size=config.kernel_size, 
                              padding=config.padding)
        self.pool = nn.MaxPool2d(kernel_size=config.pool_size)
        self.flatten = nn.Flatten()
        
        fc_input_size = self._calculate_fc_input_size(config.window_size, 
                                                     num_mel_bins)
        
        self.fc1 = nn.Linear(fc_input_size, config.hidden_size)
        self.fc_mean = nn.Linear(config.hidden_size, config.embedding_size)
        self.fc_log_var = nn.Linear(config.hidden_size, config.embedding_size)

    def _calculate_fc_input_size(self, window_size: int, num_mel_bins: int) -> int:
        """Calculate the size for the fully connected layer."""
        with torch.no_grad():
            x = torch.zeros(1, 1, window_size, num_mel_bins)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            return x.numel()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the MelEncoder."""
        if x.dim() != 3:
            raise ValueError("Input tensor must have 3 dimensions")
            
        x = x.unsqueeze(1)
        x = self._encode(x)
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)
        sample = self.reparameterize(z_mean, z_log_var)
        return sample, z_mean, z_log_var

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input tensor."""
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        return F.relu(self.fc1(x))

    def reparameterize(self, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from the latent distribution."""
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return eps * std + z_mean
    


class EnergyMapEncoder(BaseEncoder):
    """Encoder for 3D Energy Map windows."""

    def __init__(self, config: EnergyMapConfig):
        """Initialize the EnergyMapEncoder."""
        super().__init__()
        self.validate_inputs(config.window_size, config.embedding_size)
        self._init_layers(config)

    def _init_layers(self, config: EnergyMapConfig) -> None:
        """Initialize neural network layers."""
        self.conv1 = nn.Conv3d(1, config.conv1_channels, 
                              kernel_size=config.kernel_size, 
                              padding=config.padding)
        self.conv2 = nn.Conv3d(config.conv1_channels, 
                              config.conv2_channels, 
                              kernel_size=config.kernel_size, 
                              padding=config.padding)
        self.pool = nn.MaxPool3d(kernel_size=config.pool_size)
        self.flatten = nn.Flatten()

        fc_input_size = self._calculate_fc_input_size(config)
        
        self.fc1 = nn.Linear(fc_input_size, config.hidden_size)
        self.fc_mean = nn.Linear(config.hidden_size, config.embedding_size)
        self.fc_log_var = nn.Linear(config.hidden_size, config.embedding_size)

    def _calculate_fc_input_size(self, config: EnergyMapConfig) -> int:
        """Calculate the size for the fully connected layer."""
        with torch.no_grad():
            x = torch.zeros(1, 1, config.window_size, config.height, config.width)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            return x.numel()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the EnergyMapEncoder."""
        if x.dim() != 4:
            raise ValueError("Input tensor must have 4 dimensions")
            
        x = x.unsqueeze(1)
        x = self._encode(x)
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)
        sample = self.reparameterize(z_mean, z_log_var)
        return sample, z_mean, z_log_var

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input tensor."""
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        return F.relu(self.fc1(x))

    def reparameterize(self, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from the latent distribution."""
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return eps * std + z_mean

class HierarchicalDecoder(nn.Module):
    """Hierarchical Decoder with Shared and Modality-Specific Layers."""

    def __init__(self, config: DecoderConfig):
        """Initialize the HierarchicalDecoder."""
        super().__init__()
        self.config = config
        self._init_shared_layers()
        self._init_mel_layers()
        self._init_energy_layers()

    def _init_shared_layers(self) -> None:
        """Initialize shared layers."""
        self.shared_fc = nn.Linear(self.config.embedding_size * 2, 
                                 self.config.hidden_size)
        self.shared_fc2 = nn.Linear(self.config.hidden_size, 
                                  self.config.shared_hidden_size)

    def _init_mel_layers(self) -> None:
        """Initialize Mel spectrogram specific layers."""
        self.mel_fc = nn.Linear(self.config.shared_hidden_size, 
                               self.config.hidden_size)
        self.mel_unflatten = nn.Unflatten(
            1, 
            (self.config.conv_channels, 
             self.config.window_size // 4, 
             self.config.num_mel_bins // 4)
        )
        self.mel_deconv1 = nn.ConvTranspose2d(self.config.conv_channels, 
                                             self.config.conv_channels // 2, 
                                             kernel_size=2, 
                                             stride=2)
        self.mel_deconv2 = nn.ConvTranspose2d(self.config.conv_channels // 2, 
                                             1, 
                                             kernel_size=2, 
                                             stride=2)

    def _init_energy_layers(self) -> None:
        """Initialize energy map specific layers."""
        self.energy_fc = nn.Linear(self.config.shared_hidden_size, 
                                 self.config.hidden_size)
        self.energy_unflatten = nn.Unflatten(
            1, 
            (self.config.conv_channels, 
             self.config.window_size // 4, 
             self.config.height // 4, 
             self.config.width // 4)
        )
        self.energy_deconv1 = nn.ConvTranspose3d(self.config.conv_channels, 
                                                self.config.conv_channels // 2, 
                                                kernel_size=2, 
                                                stride=2)
        self.energy_deconv2 = nn.ConvTranspose3d(self.config.conv_channels // 2, 
                                                1, 
                                                kernel_size=2, 
                                                stride=2)

    def _decode_mel(self, shared: torch.Tensor) -> torch.Tensor:
        """Decode Mel spectrogram from shared representation."""
        mel = F.relu(self.mel_fc(shared))
        mel = self.mel_unflatten(mel)
        mel = F.relu(self.mel_deconv1(mel))
        mel = self.mel_deconv2(mel)
        return mel.squeeze(1)

    def _decode_energy(self, shared: torch.Tensor) -> torch.Tensor:
        """Decode energy map from shared representation."""
        energy = F.relu(self.energy_fc(shared))
        energy = self.energy_unflatten(energy)
        energy = F.relu(self.energy_deconv1(energy))
        energy = self.energy_deconv2(energy)
        return energy.squeeze(1)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the HierarchicalDecoder."""
        if z.dim() != 2:
            raise ValueError("Input tensor must have 2 dimensions")

        shared = F.relu(self.shared_fc(z))
        shared = F.relu(self.shared_fc2(shared))

        mel_recon = self._decode_mel(shared)
        energy_recon = self._decode_energy(shared)

        return mel_recon, energy_recon
    


class MultiModalCVAE(nn.Module):
    """Multi-Modal Conditional Variational Autoencoder with Separate Encoders."""

    def __init__(self, config: MultiModalConfig):
        """Initialize the MultiModalCVAE."""
        super().__init__()
        self._validate_config(config)
        self.config = config
        
        self.mel_encoder = MelEncoder(
            EncoderConfig(
                window_size=config.window_size,
                embedding_size=config.embedding_size,
                hidden_size=config.hidden_size
            ),
            num_mel_bins=config.num_mel_bins
        )
        
        self.energy_encoder = EnergyMapEncoder(
            EnergyMapConfig(
                window_size=config.window_size,
                height=config.height,
                width=config.width,
                embedding_size=config.embedding_size,
                hidden_size=config.hidden_size
            )
        )
        
        self.decoder = HierarchicalDecoder(
            DecoderConfig(
                window_size=config.window_size,
                num_mel_bins=config.num_mel_bins,
                height=config.height,
                width=config.width,
                embedding_size=config.embedding_size,
                hidden_size=config.hidden_size
            )
        )

    @staticmethod
    def _validate_config(config: MultiModalConfig) -> None:
        """Validate configuration parameters."""
        if any(param <= 0 for param in [
            config.window_size, 
            config.num_mel_bins, 
            config.height, 
            config.width, 
            config.embedding_size
        ]):
            raise ValueError("All dimensions must be positive integers")

    def encode(
        self, 
        mel_window: torch.Tensor, 
        energy_window: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode both modalities."""
        mel_z, mel_mu, mel_log_var = self.mel_encoder(mel_window)
        energy_z, energy_mu, energy_log_var = self.energy_encoder(energy_window)
        
        z = torch.cat((mel_z, energy_z), dim=1)
        z_mean = torch.cat((mel_mu, energy_mu), dim=1)
        z_log_var = torch.cat((mel_log_var, energy_log_var), dim=1)
        
        return z, z_mean, z_log_var, mel_z

    def forward(
        self, 
        mel_window: torch.Tensor, 
        energy_window: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the MultiModalCVAE."""
        self._validate_input_shapes(mel_window, energy_window)
        
        z, z_mean, z_log_var, _ = self.encode(mel_window, energy_window)
        mel_recon, energy_recon = self.decoder(z)
        
        return mel_recon, energy_recon, z_mean, z_log_var

    def _validate_input_shapes(
        self, 
        mel_window: torch.Tensor, 
        energy_window: torch.Tensor
    ) -> None:
        """Validate input tensor shapes."""
        if mel_window.dim() != 3:
            raise ValueError("Mel window must have 3 dimensions")
        if energy_window.dim() != 4:
            raise ValueError("Energy window must have 4 dimensions")

class SequencePredictor(nn.Module):
    """LSTM-based Sequence Predictor for Anomaly Detection."""

    def __init__(self, config: SequencePredictorConfig):
        """Initialize the SequencePredictor."""
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
        """Validate configuration parameters."""
        if config.num_layers <= 0:
            raise ValueError("Number of layers must be positive")
        if not 0 <= config.dropout_rate <= 1:
            raise ValueError("Dropout rate must be between 0 and 1")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the SequencePredictor."""
        self._validate_input(x)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

    def _validate_input(self, x: torch.Tensor) -> None:
        """Validate input tensor."""
        if x.dim() != 3:
            raise ValueError("Input must have 3 dimensions (batch, sequence, features)")
        if x.size(2) != self.config.input_size:
            raise ValueError(f"Expected input size {self.config.input_size}, got {x.size(2)}")

class HybridModel(nn.Module):
    """Unified Model for CVAE and LSTM-based Anomaly Detection."""

    def __init__(self, config: HybridModelConfig):
        """Initialize the HybridModel."""
        super().__init__()
        self._validate_config(config)
        self.config = config
        
        self.cvae = MultiModalCVAE(config.multimodal_config)
        self.sequence_predictor = SequencePredictor(config.sequence_config)
        self.sequence_length = config.sequence_length

    @staticmethod
    def _validate_config(config: HybridModelConfig) -> None:
        """Validate configuration parameters."""
        if config.sequence_length <= 0:
            raise ValueError("Sequence length must be positive")

    def _prepare_sequences(
        self, 
        latent_sequence: torch.Tensor,
        num_windows: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare sequences for LSTM input and target."""
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

        return (
            torch.cat(input_sequences, dim=1),
            torch.cat(target_latents, dim=1)
        )

    def _process_windows(
        self, 
        mel_windows: torch.Tensor, 
        energy_windows: torch.Tensor
    ) -> torch.Tensor:
        """Process windows through CVAE encoder."""
        batch_size, num_windows = mel_windows.shape[:2]
        latent_vectors = []

        for i in range(num_windows):
            mel_window = mel_windows[:, i, :, :]
            energy_window = energy_windows[:, i, :, :, :]
            _, _, mu, _ = self.cvae(mel_window, energy_window)
            latent_vectors.append(mu.unsqueeze(1))

        return torch.cat(latent_vectors, dim=1)

    def forward(
        self, 
        mel_windows: torch.Tensor, 
        energy_windows: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the HybridModel."""
        self._validate_input_shapes(mel_windows, energy_windows)
        
        batch_size, num_windows = mel_windows.shape[:2]
        latent_sequence = self._process_windows(mel_windows, energy_windows)
        
        input_sequences, target_latents = self._prepare_sequences(
            latent_sequence, 
            num_windows
        )
        
        # Reshape for LSTM processing
        lstm_input = self._reshape_for_lstm(input_sequences)
        predicted_latent = self.sequence_predictor(lstm_input)
        
        # Generate reconstructions
        mel_recon, energy_recon = self._generate_reconstructions(
            predicted_latent, 
            batch_size
        )
        
        # Get actual windows
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

    def _validate_input_shapes(
        self, 
        mel_windows: torch.Tensor, 
        energy_windows: torch.Tensor
    ) -> None:
        """Validate input tensor shapes."""
        if mel_windows.dim() != 4:
            raise ValueError("Mel windows must have 4 dimensions")
        if energy_windows.dim() != 5:
            raise ValueError("Energy windows must have 5 dimensions")

    def _reshape_for_lstm(self, input_sequences: torch.Tensor) -> torch.Tensor:
        """Reshape input sequences for LSTM processing."""
        return input_sequences.view(
            -1, 
            self.sequence_length, 
            self.config.sequence_config.input_size
        )

    def _generate_reconstructions(
        self, 
        predicted_latent: torch.Tensor,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate reconstructions from predicted latent vectors."""
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

    def _reshape_reconstructions(
        self, 
        mel_recon: torch.Tensor,
        energy_recon: torch.Tensor,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reshape reconstruction tensors to match expected output format."""
        num_sequences = mel_recon.size(0) // batch_size
        return (
            mel_recon.view(batch_size, num_sequences, -1, mel_recon.size(-1)),
            energy_recon.view(batch_size, num_sequences, -1, 
                            energy_recon.size(-2), energy_recon.size(-1))
        )