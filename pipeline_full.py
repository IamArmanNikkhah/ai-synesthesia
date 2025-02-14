#!/usr/bin/env python3
"""
Full Pipeline: Extract audio from .mp4 files, compute mel spectrograms,
and generate spherical power maps.

Usage:
    python pipeline_full.py <input_mp4_dir> <wav_output_dir> <sph_output_dir>
         [--sr SR] [--n_fft N_FFT] [--n_mels N_MELS] [--time_interval TIME_INTERVAL]
         [--angular_res ANGULAR_RES] [--ambi_order AMBI_ORDER]
         [--ordering {ACN,FuMa}] [--normalization {SN3D,N3D}]
"""

import argparse
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List

# Updated import paths based on project structure
from utils.AudioExctractor.extract_audio import process_directory
from utils.Mel-SpectrogramGenerator.melspectogramer import process_multiple_files, MelSpectrogram
from utils.SpatialAudioMapper.sph_power_map import generate_spherical_power_map

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_spherical_for_file(
    wav_file: Path,
    sph_output_dir: Path,
    angular_res: float,
    ambi_order: int,
    ordering: str,
    normalization: str,
) -> None:
    """
    Process a single WAV file to generate spherical power map outputs.
    The outputs include a video, CSV, and Pickle file.
    """
    output_video = sph_output_dir / f"{wav_file.stem}_sph_power_map.mp4"
    csv_output = sph_output_dir / f"{wav_file.stem}_sph_power_map.csv"
    pickle_output = sph_output_dir / f"{wav_file.stem}_sph_power_map.pkl"
    logger.info(f"Starting spherical power map extraction for {wav_file.name}")
    
    generate_spherical_power_map(
        input_wav=str(wav_file),
        output_video=str(output_video),
        csv_output=str(csv_output),
        pickle_output=str(pickle_output),
        angular_res=angular_res,
        ambi_order=ambi_order,
        ordering=ordering,
        normalization=normalization,
    )
    
    logger.info(f"Completed spherical power map extraction for {wav_file.name}")


def run_spherical_extraction(
    wav_files: List[Path],
    sph_output_dir: Path,
    angular_res: float,
    ambi_order: int,
    ordering: str,
    normalization: str,
) -> None:
    """
    Process all WAV files to extract spherical power maps concurrently using ThreadPoolExecutor.
    """
    sph_output_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(
                process_spherical_for_file,
                wav_file,
                sph_output_dir,
                angular_res,
                ambi_order,
                ordering,
                normalization,
            ): wav_file
            for wav_file in wav_files
        }
        for future in as_completed(futures):
            wav_file = futures[future]
            try:
                future.result()
            except Exception as exc:
                logger.error(
                    f"Error processing spherical power map for {wav_file.name}: {exc}"
                )


async def run_full_pipeline(
    input_mp4_dir: Path,
    wav_output_dir: Path,
    sph_output_dir: Path,
    sr: int,
    n_fft: int,
    n_mels: int,
    time_interval: Optional[float],
    angular_res: float,
    ambi_order: int,
    ordering: str,
    normalization: str,
) -> None:
    """
    Run the complete processing pipeline:
      1. Extract .wav files from .mp4 files.
      2. Compute mel spectrograms for each .wav file.
      3. Generate spherical power maps for each .wav file.
    """
    logger.info("Starting audio extraction from .mp4 files...")
    await process_directory(input_mp4_dir, wav_output_dir)
    logger.info("Audio extraction completed.")

    # Gather all extracted .wav files
    wav_files = sorted(wav_output_dir.glob("*.wav"))
    if not wav_files:
        logger.error("No .wav files found after extraction.")
        return

    # Step 2: Process mel spectrogram extraction
    audio_paths = [str(wav) for wav in wav_files]
    logger.info(f"Processing mel spectrogram extraction for {len(audio_paths)} files...")
    _ = process_multiple_files(
        audio_paths,
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        time_interval=time_interval,
    )
    logger.info("Mel spectrogram extraction completed.")

    # Step 3: Process spherical power map extraction concurrently
    logger.info("Processing spherical power map extraction for WAV files...")
    run_spherical_extraction(
        wav_files, sph_output_dir, angular_res, ambi_order, ordering, normalization
    )
    logger.info("Spherical power map extraction completed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full Pipeline: Extract audio, compute mel spectrograms, and generate spherical power maps."
    )
    parser.add_argument("input_mp4_dir", type=str, help="Directory containing .mp4 files.")
    parser.add_argument("wav_output_dir", type=str, help="Directory to store extracted .wav files.")
    parser.add_argument("sph_output_dir", type=str, help="Directory to store spherical power map outputs.")
    # Mel-spectrogram parameters
    parser.add_argument("--sr", type=int, default=22050, help="Sampling rate for audio and mel extraction.")
    parser.add_argument("--n_fft", type=int, default=2048, help="Number of FFT components for mel extraction.")
    parser.add_argument("--n_mels", type=int, default=128, help="Number of Mel bands.")
    parser.add_argument("--time_interval", type=float, default=None, help="Time interval for mel spectrogram sampling.")
    # Spherical power map parameters
    parser.add_argument("--angular_res", type=float, default=2.0, help="Angular resolution for spherical grid (degrees).")
    parser.add_argument("--ambi_order", type=int, default=1, help="Ambisonic order.")
    parser.add_argument("--ordering", type=str, default="ACN", choices=["ACN", "FuMa"], help="Ambisonic channel ordering.")
    parser.add_argument("--normalization", type=str, default="SN3D", choices=["SN3D", "N3D"], help="Ambisonic normalization scheme.")
    args = parser.parse_args()

    input_mp4_dir = Path(args.input_mp4_dir)
    wav_output_dir = Path(args.wav_output_dir)
    sph_output_dir = Path(args.sph_output_dir)

    asyncio.run(
        run_full_pipeline(
            input_mp4_dir,
            wav_output_dir,
            sph_output_dir,
            args.sr,
            args.n_fft,
            args.n_mels,
            args.time_interval,
            args.angular_res,
            args.ambi_order,
            args.ordering,
            args.normalization,
        )
    )


if __name__ == "__main__":
    main()
