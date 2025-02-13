#!/usr/bin/env python3
"""
Script to split Mel spectrogram and energy map data into training and testing sets based on specified index ranges.

Usage:
    python split_dataset.py \
        --mel_pickle_path path/to/mel_data.pkl \
        --energy_pickle_path path/to/energy_data.pkl \
        --train_mel_output path/to/train_mel.pkl \
        --train_energy_output path/to/train_energy.pkl \
        --test_mel_output path/to/test_mel.pkl \
        --test_energy_output path/to/test_energy.pkl \
        --train_ranges "1:10,30:500,612:700"

Arguments:
    --mel_pickle_path     Path to the input Mel spectrogram pickle file.
    --energy_pickle_path  Path to the input energy map pickle file.
    --train_mel_output    Path to save the training Mel spectrogram pickle file.
    --train_energy_output Path to save the training energy map pickle file.
    --test_mel_output     Path to save the testing Mel spectrogram pickle file.
    --test_energy_output  Path to save the testing energy map pickle file.
    --train_ranges        Comma-separated list of index ranges for training (e.g., "1:10,30:500,612:700").

Notes:
    - Index ranges are zero-based and follow Python's slice semantics (start inclusive, end exclusive).
    - Ensure that the specified ranges do not exceed the dataset size.
    - Overlapping ranges are allowed but will be handled by using a set for training indices.
"""

import argparse
import logging
import os
import pickle
from typing import List, Set, Tuple


def parse_ranges(range_str: str, max_index: int) -> Set[int]:
    """
    Parse a string of index ranges and return a set of indices.

    Args:
        range_str (str): Comma-separated string of ranges (e.g., "1:10,30:500,612:700").
        max_index (int): Maximum valid index (exclusive).

    Returns:
        Set[int]: Set of indices included in the specified ranges.

    Raises:
        ValueError: If the range format is incorrect or indices are out of bounds.
    """
    indices = set()
    ranges = range_str.split(',')
    for r in ranges:
        try:
            start_str, end_str = r.split(':')
            start = int(start_str)
            end = int(end_str)
            if start < 0 or end > max_index:
                raise ValueError(
                    f"Range {r} is out of bounds. Valid indices are from 0 to {max_index - 1}."
                )
            if start >= end:
                raise ValueError(
                    f"Invalid range {r}: start index must be less than end index."
                )
            indices.update(range(start, end))
        except ValueError as ve:
            raise ValueError(f"Invalid range format '{r}': {ve}")
        except Exception as e:
            raise ValueError(f"Error parsing range '{r}': {e}")
    return indices


def split_data(
    mel_data: List, energy_data: List, train_indices: Set[int]
) -> Tuple[List, List, List, List]:
    """
    Split the Mel spectrogram and energy map data into training and testing sets.

    Args:
        mel_data (List): List of Mel spectrogram frames.
        energy_data (List): List of energy map frames.
        train_indices (Set[int]): Set of indices to include in the training set.

    Returns:
        Tuple[List, List, List, List]:
            - train_mel: List of training Mel spectrogram frames.
            - train_energy: List of training energy map frames.
            - test_mel: List of testing Mel spectrogram frames.
            - test_energy: List of testing energy map frames.
    """
    train_mel = [mel_data[i] for i in train_indices]
    train_energy = [energy_data[i] for i in train_indices]

    all_indices = set(range(len(mel_data)))
    test_indices = all_indices - train_indices

    test_mel = [mel_data[i] for i in sorted(test_indices)]
    test_energy = [energy_data[i] for i in sorted(test_indices)]

    return train_mel, train_energy, test_mel, test_energy


def save_pickle(data: List, file_path: str):
    """
    Save data to a pickle file.

    Args:
        data (List): Data to be pickled.
        file_path (str): Destination file path.

    Raises:
        IOError: If the file cannot be written.
    """
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"Successfully saved data to {file_path}")
    except Exception as e:
        raise IOError(f"Failed to save data to {file_path}: {e}")


def load_pickle(file_path: str) -> List:
    """
    Load data from a pickle file.

    Args:
        file_path (str): Path to the pickle file.

    Returns:
        List: Loaded data.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the file cannot be read or is corrupted.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Pickle file not found: {file_path}")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logging.info(f"Successfully loaded data from {file_path}")
        return data
    except Exception as e:
        raise IOError(f"Failed to load data from {file_path}: {e}")


def validate_data_lengths(mel_data: List, energy_data: List):
    """
    Validate that Mel spectrogram and energy map data have the same length.

    Args:
        mel_data (List): Mel spectrogram data.
        energy_data (List): Energy map data.

    Raises:
        ValueError: If the lengths of the datasets do not match.
    """
    if len(mel_data) != len(energy_data):
        raise ValueError(
            f"Mismatch in data lengths: Mel data has {len(mel_data)} frames, "
            f"but energy data has {len(energy_data)} frames."
        )


def main():
    """Main function to parse arguments and split the dataset."""
    parser = argparse.ArgumentParser(
        description="Split Mel spectrogram and energy map data into training and testing sets based on specified index ranges."
    )
    parser.add_argument(
        "--mel_pickle_path",
        type=str,
        required=True,
        help="Path to the input Mel spectrogram pickle file.",
    )
    parser.add_argument(
        "--energy_pickle_path",
        type=str,
        required=True,
        help="Path to the input energy map pickle file.",
    )
    parser.add_argument(
        "--train_mel_output",
        type=str,
        required=True,
        help="Path to save the training Mel spectrogram pickle file.",
    )
    parser.add_argument(
        "--train_energy_output",
        type=str,
        required=True,
        help="Path to save the training energy map pickle file.",
    )
    parser.add_argument(
        "--test_mel_output",
        type=str,
        required=True,
        help="Path to save the testing Mel spectrogram pickle file.",
    )
    parser.add_argument(
        "--test_energy_output",
        type=str,
        required=True,
        help="Path to save the testing energy map pickle file.",
    )
    parser.add_argument(
        "--train_ranges",
        type=str,
        required=True,
        help='Comma-separated list of index ranges for training (e.g., "1:10,30:500,612:700").',
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        # Load data from pickle files
        mel_data = load_pickle(args.mel_pickle_path)
        energy_data = load_pickle(args.energy_pickle_path)

        # Validate data lengths
        validate_data_lengths(mel_data, energy_data)
        total_frames = len(mel_data)
        logging.info(f"Total number of frames in dataset: {total_frames}")

        # Parse training index ranges
        train_indices = parse_ranges(args.train_ranges, total_frames)
        logging.info(f"Number of training samples: {len(train_indices)}")

        # Split data into training and testing sets
        train_mel, train_energy, test_mel, test_energy = split_data(
            mel_data, energy_data, train_indices
        )
        logging.info(f"Number of testing samples: {len(test_mel)}")

        # Save split data to pickle files
        save_pickle(train_mel, args.train_mel_output)
        save_pickle(train_energy, args.train_energy_output)
        save_pickle(test_mel, args.test_mel_output)
        save_pickle(test_energy, args.test_energy_output)

        logging.info("Dataset splitting completed successfully.")

    except Exception as e:
        logging.error(f"An error occurred during dataset splitting: {e}")
        exit(1)


if __name__ == "__main__":
    main()
