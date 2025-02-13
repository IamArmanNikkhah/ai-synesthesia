#!/usr/bin/env python3

import argparse
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class MelSpectrogram:
    audio_path: str
    mel_spectrogram: np.ndarray  # Shape: (n_mels, n_frames)
    sr: int
    n_fft: int
    hop_length: int
    n_mels: int

    def plot(self) -> plt.Figure:
        """Plot the Mel spectrogram."""
        M_dB = librosa.power_to_db(self.mel_spectrogram, ref=np.max)
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(
            M_dB,
            x_axis='time',
            y_axis='mel',
            sr=self.sr,
            hop_length=self.hop_length,
            ax=ax
        )
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set(
            title=f'Mel-frequency spectrogram for {os.path.basename(self.audio_path)}'
        )
        return fig

    def save_to_csv(self, output_path: str, times: np.ndarray) -> None:
        """
        Save the Mel spectrogram data to a CSV file with frequencies as columns
        and times as rows.
        """
        # Calculate the Mel frequencies
        mel_frequencies = librosa.mel_frequencies(
            n_mels=self.n_mels,
            fmax=self.sr / 2
        )

        # Transpose the spectrogram so that frequencies are columns
        data_transposed = self.mel_spectrogram.T

        # Create a DataFrame with times as index and frequencies as columns
        df = pd.DataFrame(
            data=data_transposed,
            index=times,
            columns=mel_frequencies
        )

        # Rename index and columns
        df.index.name = 'Time (s)'
        df.columns.name = 'Mel Frequency (Hz)'

        # Save to CSV
        df.to_csv(output_path, float_format='%.6f')

    def save_to_pickle(self, output_path: str, times: np.ndarray) -> Tuple[str, Tuple[int, int]]:
        """
        Save the Mel spectrogram data to a pickle file.
        The data is stored as a 2D NumPy array with shape (n_frames, n_mels),
        where n_frames is the number of time steps and n_mels is the number of
        Mel frequency bands.

        Args:
            output_path: Path where the pickle file will be saved
            times: Array of time points (not used in pickle storage)

        Returns:
            A tuple containing the output path and the shape of the saved data.
        """
        # Transpose the spectrogram so that time steps are along the first axis
        data_transposed = self.mel_spectrogram.T  # Shape: (n_frames, n_mels)

        # Save the transposed data array to a pickle file
        with open(output_path, 'wb') as f:
            pickle.dump(data_transposed, f)

        return output_path, data_transposed.shape


def extract_mel_spectrogram(
    audio_path: str,
    sr: int = 22050,
    n_fft: int = 2048,
    n_mels: int = 128,
    time_interval: Optional[float] = None
) -> Tuple[MelSpectrogram, np.ndarray]:
    """
    Extracts a Mel spectrogram from a WAV audio file and returns a MelSpectrogram object.
    If time_interval is provided, the spectrogram is sampled at those intervals.
    """
    y, sr = librosa.load(audio_path, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr)

    # Set hop_length to ensure sufficient time resolution
    if time_interval is not None:
        # Set hop_length to ensure frames are available at desired intervals
        hop_length = max(1, int(sr * time_interval / 2))
    else:
        hop_length = 512  # Default value

    # Compute the Mel spectrogram
    M = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    # Get the times of each frame
    frame_times = librosa.frames_to_time(
        np.arange(M.shape[1]),
        sr=sr,
        hop_length=hop_length,
        n_fft=n_fft
    )

    if time_interval is not None:
        times_desired = np.arange(0, duration, time_interval)
        # Find the indices of the closest frames to the desired times
        indices = np.searchsorted(frame_times, times_desired)
        indices = np.clip(indices, 0, M.shape[1] - 1)
        M = M[:, indices]
        frame_times = frame_times[indices]  # Update frame times to match sampled data

    return MelSpectrogram(audio_path, M, sr, n_fft, hop_length, n_mels), frame_times


def process_multiple_files(
    audio_paths: List[str],
    sr: int = 22050,
    n_fft: int = 2048,
    n_mels: int = 128,
    time_interval: Optional[float] = None
) -> Dict[str, Tuple[MelSpectrogram, np.ndarray]]:
    """
    Processes multiple audio files in parallel, extracting Mel spectrograms for each.
    """
    def process_single_file(audio_path: str):
        mel_spectrogram, times = extract_mel_spectrogram(
            audio_path,
            sr=sr,
            n_fft=n_fft,
            n_mels=n_mels,
            time_interval=time_interval
        )
        return audio_path, mel_spectrogram, times

    results = {}
    with ThreadPoolExecutor() as executor:
        future_to_path = {
            executor.submit(process_single_file, path): path for path in audio_paths
        }
        for future in as_completed(future_to_path):
            audio_path = future_to_path[future]
            try:
                _, mel_spectrogram, times = future.result()
                results[audio_path] = (mel_spectrogram, times)
            except Exception as exc:
                print(f'{audio_path} generated an exception: {exc}')
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract and save Mel spectrograms from audio files."
    )
    parser.add_argument(
        'audio_files',
        nargs='+',
        help='List of audio files to process.'
    )
    parser.add_argument(
        '--sr',
        type=int,
        default=22050,
        help='Sampling rate for audio. Default is 22050 Hz.'
    )
    parser.add_argument(
        '--n_fft',
        type=int,
        default=2048,
        help='Number of FFT components. Default is 2048.'
    )
    parser.add_argument(
        '--n_mels',
        type=int,
        default=128,
        help='Number of Mel bands to generate. Default is 128.'
    )
    parser.add_argument(
        '--time_interval',
        type=float,
        default=None,
        help='Time interval in seconds for sampling the spectrogram.'
    )

    args = parser.parse_args()

    # Process multiple files in parallel
    spectrogram_data = process_multiple_files(
        args.audio_files,
        sr=args.sr,
        n_fft=args.n_fft,
        n_mels=args.n_mels,
        time_interval=args.time_interval
    )

    # Save each figure, CSV, and pickle, and print Mel spectrogram shape for each file
    for audio_path, (mel_spectrogram, times) in spectrogram_data.items():
        # Plot and save the figure
        fig = mel_spectrogram.plot()
        output_image_path = os.path.splitext(audio_path)[0] + '_mel_spectrogram.png'
        fig.savefig(output_image_path)
        plt.close(fig)  # Close the figure to free memory

        # Save the Mel spectrogram data to CSV with transposed format
        output_csv_path = os.path.splitext(audio_path)[0] + '_mel_spectrogram.csv'
        mel_spectrogram.save_to_csv(output_csv_path, times)

        # Save the Mel spectrogram data to pickle format and get the shape
        output_pickle_path = os.path.splitext(audio_path)[0] + '_mel_spectrogram.pkl'
        pickle_output_path, pickle_shape = mel_spectrogram.save_to_pickle(output_pickle_path, times)

        print(
            f"Processed {audio_path}.\n"
            f"Mel spectrogram shape: {mel_spectrogram.mel_spectrogram.shape}.\n"
            f"Saved figure as {output_image_path},\n"
            f"data as {output_csv_path},\n"
            f"and pickle as {output_pickle_path}.\n"
            f"Pickle data shape: {pickle_shape}.\n"
        )


if __name__ == "__main__":
    main()
