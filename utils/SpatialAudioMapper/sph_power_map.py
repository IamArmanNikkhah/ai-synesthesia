import argparse
import numpy as np
import pandas as pd  # For CSV handling
import pickle        # For Pickle handling
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cv2

# Import custom modules
from .video import VideoWriter  # For video output
from .decoder import AmbiDecoder  # Ambisonic decoder
from .position import Position          # Position representation for grid points
from .ambi_format import AmbiFormat      # Ambisonic format class


class SphericalAmbisonicsVisualizer:
    """
    Visualizes spherical power maps from Ambisonic data.
    """

    def __init__(
        self,
        data: np.ndarray,
        sample_rate: int,
        angular_res: float = 2.0,
        window_size: float = 0.1,
        ambi_order: int = 1,
        ordering: str = "ACN",
        normalization: str = "SN3D",
    ) -> None:
        """
        Initialize the SphericalAmbisonicsVisualizer.

        :param data: Ambisonic data (NumPy array)
        :param sample_rate: Sample rate of the audio data
        :param angular_res: Angular resolution for the spherical grid
        :param window_size: Size of the time window for frame generation (seconds)
        :param ambi_order: Order of the Ambisonic data
        :param ordering: Ambisonic channel ordering ('ACN' or 'FuMa')
        :param normalization: Ambisonic normalization scheme ('SN3D' or 'N3D')
        """
        self.data = data
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_frames = int(window_size * sample_rate)
        self.total_frames = int(np.ceil(data.shape[0] / self.window_frames))
        self.angular_res = angular_res

        # Create AmbiFormat instance
        self.ambi_format = AmbiFormat(ambi_order, ordering, normalization)

        # Validate data shape
        expected_channels = self.ambi_format.num_channels
        if data.shape[1] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} channels for order {self.ambi_format.order}, "
                f"but got {data.shape[1]}"
            )

        # Generate the spherical grid points
        (
            self.phi_grid,
            self.nu_grid,
            self.grid_positions,
        ) = self._generate_spherical_grid(angular_res)

        # Initialize the Ambisonic decoder
        self.decoder = AmbiDecoder(
            self.grid_positions, ambi_format=self.ambi_format, method="projection"
        )

    def _generate_spherical_grid(self, angular_res: float):
        """
        Generate a spherical grid of points with a given angular resolution.

        :param angular_res: Angular resolution in degrees
        :return: phi_grid, nu_grid, list of Position objects representing the grid points on a sphere
        """
        phi_vals = np.radians(np.arange(0, 360, angular_res))  # Azimuth in radians
        nu_vals = np.radians(
            np.arange(-90, 90 + angular_res, angular_res)
        )  # Elevation in radians

        phi_grid, nu_grid = np.meshgrid(phi_vals, nu_vals)

        # Create list of Position objects maintaining the 2D grid structure
        grid_positions = [
            Position(phi, nu, 1, "polar")
            for phi_row, nu_row in zip(phi_grid, nu_grid)
            for phi, nu in zip(phi_row, nu_row)
        ]

        return phi_grid, nu_grid, grid_positions

    def generate_power_map(self):
        """
        Generator function to create spherical power maps frame by frame.

        :return: Generator yielding frames as 2D NumPy arrays
        """
        rms_energies = []
        max_rms_energy = 0.0

        for i in range(self.total_frames):
            start_idx = i * self.window_frames
            end_idx = min(start_idx + self.window_frames, self.data.shape[0])

            # Extract the current chunk of ambisonic data
            ambi_chunk = self.data[start_idx:end_idx, :]

            # Handle cases where the chunk is empty
            if ambi_chunk.shape[0] == 0:
                continue

            # Decode the ambisonic chunk
            decoded_signals = self.decoder.decode(ambi_chunk)

            # Compute the RMS energy for each point on the sphere
            rms_energy = np.sqrt(np.mean(np.square(decoded_signals), axis=0))
            rms_energies.append(rms_energy)

            max_rms_energy = max(max_rms_energy, np.max(rms_energy))

        # Now generate normalized frames
        for rms_energy in rms_energies:
            # Normalize the frame for visualization
            frame = rms_energy / max_rms_energy

            # Reshape the frame into 2D array
            frame_2d = frame.reshape(self.nu_grid.shape)

            yield frame_2d


def generate_spherical_power_map(
    input_wav: str,
    output_video: str,
    csv_output: str,
    pickle_output: str,
    angular_res: float = 2.0,
    ambi_order: int = 1,
    ordering: str = "ACN",
    normalization: str = "SN3D",
) -> None:
    """
    Generates a spherical power map from an ambisonic .wav file and saves it as a video, CSV, and Pickle file.

    :param input_wav: Path to input .wav file containing ambisonic audio
    :param output_video: Path to output video file to save the spherical power map
    :param csv_output: Path to output CSV file to save the power map data
    :param pickle_output: Path to output Pickle file to save the power map data
    :param angular_res: Angular resolution for the spherical grid (degrees)
    :param ambi_order: Order of the Ambisonic data
    :param ordering: Ambisonic channel ordering ('ACN' or 'FuMa')
    :param normalization: Ambisonic normalization scheme ('SN3D' or 'N3D')
    """
    # Load the ambisonic wav file
    sample_rate, data = wavfile.read(input_wav)

    # Ensure data is in the correct format
    data = data.astype(np.float32)

    # Optional normalization if data exceeds expected ranges
    max_val = np.max(np.abs(data))
    if max_val > 1.0:
        data /= max_val

    # Initialize the visualizer
    visualizer = SphericalAmbisonicsVisualizer(
        data,
        sample_rate,
        angular_res,
        ambi_order=ambi_order,
        ordering=ordering,
        normalization=normalization,
    )

    # Set desired video dimensions divisible by 16
    video_width = 1920  # Divisible by 16
    video_height = 1088  # Adjusted to be divisible by 16

    # Initialize the video writer
    video_fps = int(1 / visualizer.window_size)
    video_writer = VideoWriter(
        output_video, video_fps=video_fps, backend="imageio", overwrite=True
    )

    # Initialize lists to collect power map data
    power_maps = []      # For CSV: list of flattened power maps
    frames_pickle = []   # For Pickle: list of 2D power maps

    # Generate frames and write them to the video file
    cmap = plt.get_cmap("inferno")
    for frame in visualizer.generate_power_map():
        # Apply the colormap and convert to 8-bit for video
        colored_frame = (cmap(frame)[:, :, :3] * 255).astype(np.uint8)

        # Resize frame to desired video dimensions
        resized_frame = cv2.resize(
            colored_frame, (video_width, video_height), interpolation=cv2.INTER_LINEAR
        )

        # Write the frame to the video
        video_writer.write_frame(resized_frame)

        # Collect the power map data for CSV and Pickle
        power_maps.append(frame.flatten())      # Flatten for CSV
        frames_pickle.append(frame.copy())      # Keep 2D for Pickle

    video_writer.close()

    # ---------------------- Export to CSV ---------------------- #

    # Convert power maps to NumPy array
    power_maps_array = np.array(power_maps)  # Shape: (num_frames, num_grid_points)

    # Generate column names based on azimuth and elevation angles
    phi_flat = visualizer.phi_grid.flatten()
    nu_flat = visualizer.nu_grid.flatten()
    phi_degrees = np.degrees(phi_flat)
    nu_degrees = np.degrees(nu_flat)
    column_names = [
        f"az_{phi:.1f}_el_{nu:.1f}" for phi, nu in zip(phi_degrees, nu_degrees)
    ]

    # Create DataFrame
    df = pd.DataFrame(power_maps_array, columns=column_names)

    # Write DataFrame to CSV file
    df.to_csv(csv_output, index=False)

    # ---------------------- Export to Pickle ---------------------- #

    # Convert list of 2D frames to a single 3D NumPy array
    frames_pickle_array = np.array(frames_pickle)  # Shape: (num_frames, height, width)

    # Save the 3D array to a Pickle file
    with open(pickle_output, 'wb') as f:
        pickle.dump(frames_pickle_array, f)

    # Compute total duration of the audio
    total_duration = data.shape[0] / sample_rate

    # Print output information
    print(f"Exported CSV to {csv_output}")
    print(f"Exported Pickle to {pickle_output}")
    print(f"Frame shape saved in Pickle: {frames_pickle_array.shape}")
    print(f"Number of frames: {frames_pickle_array.shape[0]}")
    print(f"Frame dimensions (height, width): ({frames_pickle_array.shape[1]}, {frames_pickle_array.shape[2]})")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Total duration of audio processed: {total_duration:.2f} seconds")


def main() -> None:
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate spherical power maps from ambisonic .wav files."
    )
    parser.add_argument("input_wav", help="Path to the input ambisonic .wav file")
    parser.add_argument("output_video", help="Path to save the output video")
    parser.add_argument("csv_output", help="Path to save the output CSV file")
    parser.add_argument("pickle_output", help="Path to save the output Pickle file")
    parser.add_argument(
        "--angular_res",
        type=float,
        default=2.0,
        help="Angular resolution for the spherical grid (degrees)",
    )
    parser.add_argument(
        "--ambi_order",
        type=int,
        default=1,
        help="Order of the Ambisonic data (default: 1)",
    )
    parser.add_argument(
        "--ordering",
        type=str,
        default="ACN",
        choices=["ACN", "FuMa"],
        help="Ambisonic channel ordering ('ACN' or 'FuMa')",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="SN3D",
        choices=["SN3D", "N3D"],
        help="Ambisonic normalization scheme ('SN3D' or 'N3D')",
    )

    args = parser.parse_args()

    # Generate the spherical power map, save it to a video, CSV, and Pickle file
    generate_spherical_power_map(
        args.input_wav,
        args.output_video,
        args.csv_output,
        args.pickle_output,
        angular_res=args.angular_res,
        ambi_order=args.ambi_order,
        ordering=args.ordering,
        normalization=args.normalization,
    )


if __name__ == "__main__":
    main()
