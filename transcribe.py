#!/usr/bin/env python3
"""
Batch file transcription using NVIDIA NeMo Parakeet.

Transcribes one or more audio files (or all files in a directory) and prints
the recognized text to stdout.

Usage:
    python transcribe.py audio.wav
    python transcribe.py ./recordings/
    python transcribe.py audio.wav --model nvidia/parakeet-tdt-0.6b-v2
"""

import argparse
import time
from pathlib import Path

import torch
import nemo.collections.asr as nemo_asr


DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v2"
SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def load_model(model_name: str = DEFAULT_MODEL):
    """
    Load the NeMo ASR model onto the best available device.

    Returns the model ready for inference.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading '{model_name}' on {device} …")

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    model = model.to(device).eval()

    print("Model ready.\n")
    return model


def collect_audio_files(path: Path) -> list[Path]:
    """
    Return a sorted list of supported audio files found at path.

    Accepts a single file or a directory (searched recursively).
    """
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_EXTENSIONS else []

    if path.is_dir():
        return sorted(
            f for f in path.rglob("*")
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    return []


def transcribe_files(model, audio_files: list[Path]) -> list[dict]:
    """
    Transcribe audio_files in a single batched call and return a list of
    {'file': str, 'transcript': str} dicts.

    NeMo handles batching internally; GPU utilisation is maximised automatically.
    """
    file_paths = [str(f) for f in audio_files]
    print(f"Transcribing {len(file_paths)} file(s) …\n")

    t0 = time.perf_counter()
    transcriptions = model.transcribe(file_paths)
    elapsed = time.perf_counter() - t0

    results = []

    for audio_file, text in zip(audio_files, transcriptions):
        entry = {"file": audio_file.name, "transcript": text.strip()}
        results.append(entry)
        print(f"  ▸ {audio_file.name}")
        print(f"    {text.strip() or '(empty)'}\n")

    print(f"Finished {len(results)} file(s) in {elapsed:.2f}s")
    return results


def main() -> None:
    """Parse CLI arguments and run batch transcription."""
    parser = argparse.ArgumentParser(
        description="Batch transcription with NVIDIA NeMo Parakeet"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Audio file or directory of audio files to transcribe",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"NeMo ASR model name (default: {DEFAULT_MODEL})",
    )

    args = parser.parse_args()

    audio_files = collect_audio_files(args.input)

    if not audio_files:
        print(f"No supported audio files found at: {args.input}")
        print(f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        return

    model = load_model(args.model)
    transcribe_files(model, audio_files)


if __name__ == "__main__":
    main()
