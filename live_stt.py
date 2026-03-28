#!/usr/bin/env python3
"""
Live Speech-to-Text using NVIDIA NeMo Parakeet + Silero VAD.

Three-stage multi-threaded pipeline:
    AudioCaptureThread  -> captures microphone audio in real-time
    VadProcessorThread  -> detects speech vs. silence with Silero VAD
    TranscriptionThread -> transcribes confirmed speech with NeMo Parakeet

Usage:
    python live_stt.py
    python live_stt.py --model nvidia/parakeet-tdt-0.6b-v2 --vad-threshold 0.6
"""

import argparse
import queue
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import nemo.collections.asr as nemo_asr


# ---------------------------------------------------------------------------
# Default configuration constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v2"
SAMPLE_RATE = 16000                                   # Hz – required by both models
CHUNK_DURATION_SEC = 0.5                              # length of each audio chunk
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_SEC)   # samples per chunk
DEFAULT_VAD_THRESHOLD = 0.5                           # Silero confidence ∈ [0, 1]
SILENCE_DURATION_SEC = 1.0                            # silence that triggers utterance end
MIN_SPEECH_DURATION_SEC = 0.4                         # discard very short detections


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    """Return a human-readable HH:MM:SS.mmm timestamp for console logs."""
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(model_name: str) -> tuple:
    """
    Load NeMo Parakeet ASR model and Silero VAD model.

    Returns (asr_model, vad_model, device).
    Prefers CUDA when available, otherwise falls back to CPU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{_ts()}] Device: {device}")

    print(f"[{_ts()}] Loading Silero VAD from torch.hub ...")
    # Silero VAD: default sample_rate=16000, threshold configurable at inference time
    vad_model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
    )
    vad_model = vad_model.to(device).eval()

    print(f"[{_ts()}] Loading NeMo ASR model '{model_name}' ...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    asr_model = asr_model.to(device).eval()

    print(f"[{_ts()}] Models loaded.\n")
    return asr_model, vad_model, device


# ---------------------------------------------------------------------------
# Thread: Audio capture
# ---------------------------------------------------------------------------

def audio_capture_thread(
    stop_event: threading.Event,
    audio_q: queue.Queue,
) -> None:
    """
    Capture mono 16-kHz audio from the default input device and enqueue chunks.

    Drops chunks silently when the queue is full to prevent pipeline stalls.
    """

    def _callback(indata, frames, time_info, status):
        if status:
            print(f"[{_ts()}] ⚠ Audio status: {status}")

        chunk = indata[:, 0].copy().astype(np.float32)

        try:
            audio_q.put_nowait(chunk)
        except queue.Full:
            pass  # drop to stay real-time

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=CHUNK_SIZE,
        callback=_callback,
    ):
        print(f"[{_ts()}] 🎤 Microphone open – speak now.  (Ctrl+C to quit)")
        while not stop_event.is_set():
            time.sleep(0.05)

    print(f"[{_ts()}] AudioCapture stopped.")


# ---------------------------------------------------------------------------
# Thread: VAD processor
# ---------------------------------------------------------------------------

def vad_processor_thread(
    stop_event: threading.Event,
    audio_q: queue.Queue,
    speech_q: queue.Queue,
    vad_model: torch.nn.Module,
    device: torch.device,
    vad_threshold: float,
) -> None:
    """
    Consume raw audio chunks, run Silero VAD on each, and assemble complete
    speech segments.  A segment ends after SILENCE_DURATION_SEC of quiet.
    """
    speech_buffer: list[np.ndarray] = []
    silence_chunks = 0
    is_speaking = False

    max_silence_chunks = int(SILENCE_DURATION_SEC / CHUNK_DURATION_SEC)
    min_speech_chunks = int(MIN_SPEECH_DURATION_SEC / CHUNK_DURATION_SEC)

    while not stop_event.is_set() or not audio_q.empty():
        try:
            chunk = audio_q.get(timeout=0.2)
        except queue.Empty:
            continue

        tensor = torch.from_numpy(chunk).unsqueeze(0).to(device)

        with torch.no_grad():
            speech_prob = vad_model(tensor, SAMPLE_RATE).item()

        if speech_prob >= vad_threshold:
            speech_buffer.append(chunk)
            silence_chunks = 0
            is_speaking = True

        elif is_speaking:
            # Include a small trailing tail so transcription sounds natural
            speech_buffer.append(chunk)
            silence_chunks += 1

            if silence_chunks >= max_silence_chunks:
                if len(speech_buffer) >= min_speech_chunks:
                    segment = np.concatenate(speech_buffer)
                    speech_q.put(segment)
                    duration = len(segment) / SAMPLE_RATE
                    print(f"[{_ts()}] VAD: segment {duration:.1f}s → transcription queue")

                speech_buffer = []
                silence_chunks = 0
                is_speaking = False

    # Flush any remaining speech when shutting down
    if len(speech_buffer) >= min_speech_chunks:
        speech_q.put(np.concatenate(speech_buffer))

    print(f"[{_ts()}] VadProcessor stopped.")


# ---------------------------------------------------------------------------
# Thread: Transcription
# ---------------------------------------------------------------------------

def transcription_thread(
    stop_event: threading.Event,
    speech_q: queue.Queue,
    asr_model,
    device: torch.device,
) -> None:
    """
    Consume confirmed speech segments, write each to a temporary WAV file,
    transcribe with NeMo Parakeet, and print the result to stdout.
    """
    while not stop_event.is_set() or not speech_q.empty():
        try:
            audio_segment = speech_q.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            # NamedTemporaryFile with delete=False so NeMo can re-open the path
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)

            sf.write(str(tmp_path), audio_segment, SAMPLE_RATE)
            transcriptions = asr_model.transcribe([str(tmp_path)])
            text = transcriptions[0].strip() if transcriptions else ""

            if text:
                print(f"[{_ts()}] 📝 {text}")

        except Exception as exc:
            print(f"[{_ts()}] Transcription error: {exc}")

        finally:
            tmp_path.unlink(missing_ok=True)

    print(f"[{_ts()}] Transcription stopped.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse CLI arguments, load models, start pipeline threads, and wait."""
    parser = argparse.ArgumentParser(
        description="Live STT — NeMo Parakeet + Silero VAD (multi-threaded)"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"NeMo ASR model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=DEFAULT_VAD_THRESHOLD,
        dest="vad_threshold",
        help=f"Silero VAD speech probability threshold 0–1 (default: {DEFAULT_VAD_THRESHOLD})",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  NVIDIA NeMo Parakeet — Live Speech-to-Text")
    print(f"  Model    : {args.model}")
    print(f"  VAD thr  : {args.vad_threshold}")
    print(f"  Device   : {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)

    asr_model, vad_model, device = load_models(args.model)

    stop_event = threading.Event()
    audio_q: queue.Queue = queue.Queue(maxsize=100)
    speech_q: queue.Queue = queue.Queue(maxsize=20)

    threads = [
        threading.Thread(
            target=audio_capture_thread,
            args=(stop_event, audio_q),
            name="AudioCapture",
            daemon=True,
        ),
        threading.Thread(
            target=vad_processor_thread,
            args=(stop_event, audio_q, speech_q, vad_model, device, args.vad_threshold),
            name="VadProcessor",
            daemon=True,
        ),
        threading.Thread(
            target=transcription_thread,
            args=(stop_event, speech_q, asr_model, device),
            name="Transcription",
            daemon=True,
        ),
    ]

    for t in threads:
        t.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print(f"\n[{_ts()}] Interrupt received – shutting down …")
        stop_event.set()
        for t in threads:
            t.join(timeout=8.0)
        print(f"[{_ts()}] Done. Goodbye!")


if __name__ == "__main__":
    main()
