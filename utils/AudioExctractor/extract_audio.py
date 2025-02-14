#!/usr/bin/env python3
"""
Module: extract_audio
Description: Extracts 4-channel Ambisonics audio from .mp4 files using ffmpeg.
             Each .mp4 file is processed asynchronously and converted into a .wav file.
Usage:
    python extract_audio.py <input_directory> <output_directory>
"""

import asyncio
import logging
from pathlib import Path
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def extract_audio(input_file: Path, output_file: Path) -> None:
    """
    Extracts audio from a single .mp4 file and writes a 4-channel WAV file.

    Args:
        input_file (Path): Path to the input .mp4 file.
        output_file (Path): Path to the output .wav file.
    
    Raises:
        RuntimeError: If ffmpeg fails to extract audio.
    """
    command: List[str] = [
        "ffmpeg",
        "-y",                # Overwrite output files without asking.
        "-i", str(input_file),
        "-vn",               # No video output.
        "-ac", "4",          # Force 4 audio channels.
        "-c:a", "pcm_s16le", # Use uncompressed PCM 16-bit little endian.
        str(output_file),
    ]
    
    logger.info(f"Starting extraction: {input_file.name} -> {output_file.name}")
    
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    
    if process.returncode != 0:
        error_message = stderr.decode().strip()
        logger.error(f"Extraction failed for {input_file}: {error_message}")
        raise RuntimeError(f"ffmpeg error for {input_file}: {error_message}")
    
    logger.info(f"Extraction successful: {output_file}")


async def process_directory(input_dir: Path, output_dir: Path) -> None:
    """
    Processes all .mp4 files in a given directory and extracts audio concurrently.

    Args:
        input_dir (Path): Directory containing .mp4 files.
        output_dir (Path): Directory to save extracted .wav files.
    """
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist or is not a directory: {input_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mp4_files: List[Path] = list(input_dir.glob("*.mp4"))
    if not mp4_files:
        logger.warning(f"No .mp4 files found in {input_dir}.")
        return
    
    tasks = []
    for file in mp4_files:
        # Ensure output file has the same stem with .wav extension
        output_file = output_dir / (file.stem + ".wav")
        tasks.append(extract_audio(file, output_file))
    
    await asyncio.gather(*tasks)


def main(input_dir: str, output_dir: str) -> None:
    """
    Entry point for the audio extraction script.

    Args:
        input_dir (str): Path to the input directory.
        output_dir (str): Path to the output directory.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    asyncio.run(process_directory(input_path, output_path))


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python extract_audio.py <input_dir> <output_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
