from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union, List
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EncoderConfig:
    """
    Configuration for the MelEncoder.
    
    Attributes:
        window_size (int): The temporal window size of the input.
        embedding_size (int): The size of the latent embedding.
        hidden_size (int): The size of the hidden layer before producing latent parameters.
        conv1_channels (int): Number of output channels for the first convolution.
        conv2_channels (int): Number of output channels for the second convolution.
        kernel_size (int): Size of the convolution kernel.
        padding (int): Padding size for the convolutions.
    """
    window_size: int
    embedding_size: int
    hidden_size: int = 512
    conv1_channels: int = 16
    conv2_channels: int = 32
    kernel_size: int = 3
    padding: int = 1


@dataclass
class EnergyMapConfig:
    """
    Configuration for the EnergyMapEncoder.
    
    Attributes:
        window_size (int): The temporal window size of the input.
        height (int): The spatial height of the input energy map.
        width (int): The spatial width of the input energy map.
        embedding_size (int): The size of the latent embedding.
        hidden_size (int): The size of the hidden layer before producing latent parameters.
        conv1_channels (int): Number of output channels for the first convolution.
        conv2_channels (int): Number of output channels for the second convolution.
        kernel_size (int): Size of the convolution kernel.
        padding (int): Padding size for the convolutions.
    """
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
    """
    Configuration for the hierarchical decoder.
    
    Note: Instead of assuming the unflatten shape is determined by dividing the
    final output size by 4, we introduce an intermediate resolution (e.g. an
    arbitrary “bottleneck” size) that is later upsampled to the target.
    """
    window_size: int         # Final output temporal dimension
    num_mel_bins: int        # Final output mel frequency dimension
    height: int              # Final output spatial height for energy map
    width: int               # Final output spatial width for energy map
    embedding_size: int      # Latent embedding size per modality
    hidden_size: int = 512   # Hidden layer size after the FC layer
    shared_hidden_size: int = 1024
    conv_channels: int = 32  # Number of channels to use in the intermediate representation
    # New optional parameters for intermediate resolution:
    mel_intermediate: tuple = (4, 4)    # (H, W) for mel branch (can be arbitrary)
    energy_intermediate: tuple = (4, 4, 4)  # (D, H, W) for energy branch



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



# ---------------------------------------------------------------------------
# MelEncoder using AdaptiveAvgPool2d
# ---------------------------------------------------------------------------

class MelEncoder(nn.Module):
    """
    Encoder for 2D Mel Spectrogram windows that uses AdaptiveAvgPool2d
    to ensure a fixed-size feature vector prior to the fully connected layers.
    """
    def __init__(self, config: EncoderConfig, num_mel_bins: int):
        """
        Initialize the MelEncoder.
        
        Args:
            config (EncoderConfig): Encoder configuration.
            num_mel_bins (int): Number of Mel frequency bins.
        """
        super().__init__()
        if config.window_size <= 0 or config.embedding_size <= 0:
            raise ValueError("Window size and embedding size must be positive integers")
        
        self.config = config
        self.num_mel_bins = num_mel_bins
        
        # Convolutional layers.
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=config.conv1_channels,
            kernel_size=config.kernel_size,
            padding=config.padding
        )
        self.conv2 = nn.Conv2d(
            in_channels=config.conv1_channels,
            out_channels=config.conv2_channels,
            kernel_size=config.kernel_size,
            padding=config.padding
        )
        
        # Adaptive average pooling to force the output to (1, 1) regardless of input size.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers.
        # After adaptive pooling, the feature map has shape (batch, conv2_channels, 1, 1)
        # so the flattened feature vector has size conv2_channels.
        self.fc1 = nn.Linear(config.conv2_channels, config.hidden_size)
        self.fc_mean = nn.Linear(config.hidden_size, config.embedding_size)
        self.fc_log_var = nn.Linear(config.hidden_size, config.embedding_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the MelEncoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, window_size, num_mel_bins).
        
        Returns:
            A tuple containing:
                - sample (torch.Tensor): The sampled latent vector.
                - z_mean (torch.Tensor): The mean of the latent distribution.
                - z_log_var (torch.Tensor): The log variance of the latent distribution.
        """
        if x.dim() != 3:
            raise ValueError("Input tensor must have 3 dimensions (batch, window_size, num_mel_bins)")
        
        # Add the channel dimension: (batch, 1, window_size, num_mel_bins)
        x = x.unsqueeze(1)
        
        # Pass through convolutional layers with ReLU activations.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Apply adaptive average pooling to obtain a fixed spatial size (1, 1).
        x = self.adaptive_pool(x)
        
        # Flatten the tensor: shape becomes (batch, conv2_channels)
        x = torch.flatten(x, start_dim=1)
        
        # Process through the fully connected layers.
        hidden = F.relu(self.fc1(x))
        z_mean = self.fc_mean(hidden)
        z_log_var = self.fc_log_var(hidden)
        
        # Sample from the latent space using the reparameterization trick.
        z = self.reparameterize(z_mean, z_log_var)
        return z, z_mean, z_log_var

    def reparameterize(self, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from the latent distribution.
        
        Args:
            z_mean (torch.Tensor): Mean of the latent distribution.
            z_log_var (torch.Tensor): Log variance of the latent distribution.
        
        Returns:
            torch.Tensor: A latent vector sample.
        """
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return eps * std + z_mean

# ---------------------------------------------------------------------------
# EnergyMapEncoder using AdaptiveAvgPool3d
# ---------------------------------------------------------------------------

class EnergyMapEncoder(nn.Module):
    """
    Encoder for 3D Energy Map windows that uses AdaptiveAvgPool3d
    to ensure a fixed-size feature vector prior to the fully connected layers.
    """
    def __init__(self, config: EnergyMapConfig):
        """
        Initialize the EnergyMapEncoder.
        
        Args:
            config (EnergyMapConfig): Configuration for the energy map encoder.
        """
        super().__init__()
        if config.window_size <= 0 or config.embedding_size <= 0:
            raise ValueError("Window size and embedding size must be positive integers")
        
        self.config = config
        
        # Convolutional layers.
        self.conv1 = nn.Conv3d(
            in_channels=1,
            out_channels=config.conv1_channels,
            kernel_size=config.kernel_size,
            padding=config.padding
        )
        self.conv2 = nn.Conv3d(
            in_channels=config.conv1_channels,
            out_channels=config.conv2_channels,
            kernel_size=config.kernel_size,
            padding=config.padding
        )
        
        # Adaptive average pooling to force the output to (1, 1, 1) regardless of input size.
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Fully connected layers.
        # After adaptive pooling, the feature map has shape (batch, conv2_channels, 1, 1, 1)
        # so the flattened feature vector has size conv2_channels.
        self.fc1 = nn.Linear(config.conv2_channels, config.hidden_size)
        self.fc_mean = nn.Linear(config.hidden_size, config.embedding_size)
        self.fc_log_var = nn.Linear(config.hidden_size, config.embedding_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the EnergyMapEncoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, window_size, height, width).
        
        Returns:
            A tuple containing:
                - sample (torch.Tensor): The sampled latent vector.
                - z_mean (torch.Tensor): The mean of the latent distribution.
                - z_log_var (torch.Tensor): The log variance of the latent distribution.
        """
        if x.dim() != 4:
            raise ValueError("Input tensor must have 4 dimensions (batch, window_size, height, width)")
        
        # Add the channel dimension: (batch, 1, window_size, height, width)
        x = x.unsqueeze(1)
        
        # Pass through convolutional layers with ReLU activations.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Apply adaptive average pooling to obtain a fixed spatio-temporal size (1, 1, 1).
        x = self.adaptive_pool(x)
        
        # Flatten the tensor: shape becomes (batch, conv2_channels)
        x = torch.flatten(x, start_dim=1)
        
        # Process through the fully connected layers.
        hidden = F.relu(self.fc1(x))
        z_mean = self.fc_mean(hidden)
        z_log_var = self.fc_log_var(hidden)
        
        # Sample from the latent space using the reparameterization trick.
        z = self.reparameterize(z_mean, z_log_var)
        return z, z_mean, z_log_var

    def reparameterize(self, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from the latent distribution.
        
        Args:
            z_mean (torch.Tensor): Mean of the latent distribution.
            z_log_var (torch.Tensor): Log variance of the latent distribution.
        
        Returns:
            torch.Tensor: A latent vector sample.
        """
        std = torch.exp(0.5 * z_log_var)
        eps = torch.randn_like(std)
        return eps * std + z_mean


# ---------------------------------------------------------------------------
#  Hierarchical Decoder
# ---------------------------------------------------------------------------

class HierarchicalDecoder(nn.Module):
    """
    Hierarchical Decoder with Shared and Modality-Specific Layers.
    
    This revised decoder decouples the required final output shape from the
    shape produced immediately by the FC layers. The decoder first maps the
    latent vector to an intermediate feature map of a chosen resolution, and then
    uses transposed convolutions and/or interpolation to achieve the final desired size.
    """
    
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config
        
        # Shared layers that combine the two latent embeddings.
        self.shared_fc = nn.Linear(config.embedding_size * 2, config.hidden_size)
        self.shared_fc2 = nn.Linear(config.hidden_size, config.shared_hidden_size)
        
        # -------------------------
        # Mel Branch Initialization
        # -------------------------
        # Instead of requiring that hidden_size = conv_channels * (window_size//4) * (num_mel_bins//4),
        # we choose an intermediate resolution (e.g. config.mel_intermediate) that is flexible.
        interm_h, interm_w = config.mel_intermediate
        self.mel_intermediate_numel = config.conv_channels * interm_h * interm_w
        
        # Map shared features to a vector that we reshape into (conv_channels, interm_h, interm_w)
        self.mel_fc = nn.Linear(config.shared_hidden_size, self.mel_intermediate_numel)
        
        # Two transposed conv layers to begin upsampling:
        self.mel_deconv1 = nn.ConvTranspose2d(
            in_channels=config.conv_channels,
            out_channels=config.conv_channels // 2,
            kernel_size=2,
            stride=2
        )
        self.mel_deconv2 = nn.ConvTranspose2d(
            in_channels=config.conv_channels // 2,
            out_channels=1,  # We want one channel for mel spectrogram reconstruction.
            kernel_size=2,
            stride=2
        )
        
        # -------------------------
        # Energy Branch Initialization
        # -------------------------
        # Similarly, choose an intermediate resolution for the energy map.
        interm_d, interm_h_e, interm_w_e = config.energy_intermediate
        self.energy_intermediate_numel = config.conv_channels * interm_d * interm_h_e * interm_w_e
        
        self.energy_fc = nn.Linear(config.shared_hidden_size, self.energy_intermediate_numel)
        
        self.energy_deconv1 = nn.ConvTranspose3d(
            in_channels=config.conv_channels,
            out_channels=config.conv_channels // 2,
            kernel_size=2,
            stride=2
        )
        self.energy_deconv2 = nn.ConvTranspose3d(
            in_channels=config.conv_channels // 2,
            out_channels=1,  # One channel for the energy map.
            kernel_size=2,
            stride=2
        )

    def _decode_mel(self, shared: torch.Tensor) -> torch.Tensor:
        """
        Decode the Mel spectrogram branch from the shared latent features.
        This function uses a learned intermediate resolution and then adjusts
        the output to the final (window_size, num_mel_bins) using interpolation.
        """
        # Map shared features to an intermediate vector.
        mel = F.relu(self.mel_fc(shared))
        batch_size = mel.size(0)
        # Reshape to (batch, conv_channels, interm_h, interm_w)
        interm_h, interm_w = self.config.mel_intermediate
        mel = mel.view(batch_size, self.config.conv_channels, interm_h, interm_w)
        
        # Upsample using transposed convolutions.
        mel = F.relu(self.mel_deconv1(mel))
        mel = self.mel_deconv2(mel)
        # At this point, the output spatial dimensions depend on the chosen intermediate size
        # and the transposed convolutions’ parameters.
        # To ensure we get exactly (window_size, num_mel_bins), use interpolation:
        target_size = (self.config.window_size, self.config.num_mel_bins)
        mel = F.interpolate(mel, size=target_size, mode='bilinear', align_corners=False)
        
        # Instead of a blind squeeze, check that the channel dimension is 1 before squeezing.
        if mel.shape[1] == 1:
            mel = mel.squeeze(1)
        return mel

    def _decode_energy(self, shared: torch.Tensor) -> torch.Tensor:
        """
        Decode the Energy map branch from the shared latent features.
        Uses an intermediate resolution and interpolation to reach the desired
        (window_size, height, width) shape.
        """
        energy = F.relu(self.energy_fc(shared))
        batch_size = energy.size(0)
        interm_d, interm_h, interm_w = self.config.energy_intermediate
        # Reshape to (batch, conv_channels, interm_d, interm_h, interm_w)
        energy = energy.view(batch_size, self.config.conv_channels, interm_d, interm_h, interm_w)
        
        energy = F.relu(self.energy_deconv1(energy))
        energy = self.energy_deconv2(energy)
        # Ensure final output has the desired spatial dimensions.
        target_size = (self.config.window_size, self.config.height, self.config.width)
        energy = F.interpolate(energy, size=target_size, mode='trilinear', align_corners=False)
        
        if energy.shape[1] == 1:
            energy = energy.squeeze(1)
        return energy

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the HierarchicalDecoder.
        
        Args:
            z (torch.Tensor): Concatenated latent vector of shape (batch, embedding_size * 2)
        
        Returns:
            A tuple containing:
                - mel_recon (torch.Tensor): Reconstructed mel spectrogram with shape
                  (batch, window_size, num_mel_bins)
                - energy_recon (torch.Tensor): Reconstructed energy map with shape
                  (batch, window_size, height, width)
        """
        if z.dim() != 2:
            raise ValueError("Input tensor must have 2 dimensions")
        
        shared = F.relu(self.shared_fc(z))
        shared = F.relu(self.shared_fc2(shared))
        
        mel_recon = self._decode_mel(shared)
        energy_recon = self._decode_energy(shared)
        
        return mel_recon, energy_recon



# ---------------------------------------------------------------------------
#  MultiModal CVAE (May need modification)
# ---------------------------------------------------------------------------    


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
