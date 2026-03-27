# NeMo Playground — Live Speech-to-Text with Parakeet + Silero VAD

A Python project that explores NVIDIA's [NeMo toolkit](https://github.com/NVIDIA/NeMo), specifically the **Parakeet TDT 0.6B** ASR model, combined with **Silero VAD** for voice activity detection and a **multi-threaded** real-time transcription pipeline.

---

## Features

| Feature | Description |
|---------|-------------|
| **Live / real-time STT** | Streams microphone audio and transcribes speech continuously |
| **Silero VAD** | Segments audio into speech / silence regions before transcription |
| **Multi-threading** | Three decoupled threads (capture → VAD → transcription) connected via queues |
| **Batch transcription** | Transcribe one file or an entire directory with a single command |
| **CUDA acceleration** | Automatically uses GPU when available; falls back to CPU |

---

## Requirements

### Hardware

- NVIDIA GPU with CUDA support (recommended: RTX 30xx or newer, ≥ 6 GB VRAM)
- Microphone (for live STT)

### Software

| Dependency | Version |
|------------|---------|
| Python | 3.10 – 3.12 |
| CUDA Toolkit | 11.8 or 12.x |
| cuDNN | Compatible with your CUDA version |
| PyTorch (CUDA build) | ≥ 2.1.0 |

---

## Setup

### 1 — Create a Python virtual environment

```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 2 — Install CUDA-enabled PyTorch

Visit https://pytorch.org/get-started/locally/ and select your OS, CUDA version, and package manager.  Example for CUDA 12.1:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3 — Install project dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `nemo_toolkit[asr]` installs a large set of transitive dependencies (including `hydra`, `omegaconf`, `webdataset`, etc.).  Initial setup may take several minutes.

### 4 — (Windows) PortAudio for sounddevice

`sounddevice` requires **PortAudio**.  On Windows the easiest path is:

```powershell
pip install pipwin
pipwin install pyaudio
```

Alternatively, install a prebuilt PortAudio DLL and ensure it is on `PATH`.

---

## Usage

### Live speech-to-text (microphone → real-time transcript)

```bash
python live_stt.py
```

Optional flags:

```
--model           NeMo model name (default: nvidia/parakeet-tdt-0.6b-v2)
--vad-threshold   Silero VAD confidence threshold 0–1 (default: 0.5)
```

Example with custom settings:

```bash
python live_stt.py --model nvidia/parakeet-tdt-0.6b-v2 --vad-threshold 0.65
```

### Batch file transcription

```bash
# Single file
python transcribe.py recording.wav

# Entire directory (recursive)
python transcribe.py ./recordings/

# Use a different model
python transcribe.py audio.wav --model nvidia/parakeet-tdt-0.6b-v2
```

Supported audio formats: `.wav`, `.flac`, `.mp3`, `.ogg`, `.m4a`

---

## Architecture — live_stt.py

```
┌─────────────────┐   raw chunks    ┌──────────────────┐   speech segs   ┌──────────────────┐
│  AudioCapture   │ ──────────────► │   VadProcessor   │ ──────────────► │  Transcription   │
│  Thread         │   queue (100)   │   Thread         │   queue (20)    │  Thread          │
│                 │                 │                  │                 │                  │
│  sounddevice    │                 │  Silero VAD      │                 │  NeMo Parakeet   │
│  16 kHz mono    │                 │  0.5s chunks     │                 │  TDT 0.6B        │
└─────────────────┘                 └──────────────────┘                 └──────────────────┘
```

### How it works

1. **AudioCapture** opens the default microphone at 16 kHz and pushes 0.5-second `float32` NumPy arrays into an `audio_queue`.
2. **VadProcessor** dequeues each chunk, runs Silero VAD inference, and accumulates chunks that exceed the configured confidence threshold.  When speech is followed by ≥ 1 second of silence the assembled segment is pushed onto `speech_queue`.
3. **Transcription** dequeues each speech segment, writes it to a temporary WAV file, passes it to `nemo_asr.models.ASRModel.transcribe()`, and prints the recognized text.

A shared `threading.Event` (`stop_event`) signals all threads to drain their queues and exit cleanly when the user presses **Ctrl+C**.

---

## Model

| Property | Value |
|----------|-------|
| Name | `nvidia/parakeet-tdt-0.6b-v2` |
| Architecture | FastConformer-TDT |
| Parameters | ~600 M |
| Languages | English |
| Sample rate | 16 000 Hz |
| Source | [NVIDIA NGC / NeMo Hub](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) |

> Check [huggingface.co/nvidia](https://huggingface.co/nvidia) or the [NeMo model card list](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/results.html) for the latest Parakeet release and update the `--model` flag accordingly.

---

## Silero VAD

[Silero VAD](https://github.com/snakers4/silero-vad) is a lightweight, high-accuracy voice activity detector loaded via `torch.hub`:

```python
vad_model, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
)
```

No extra package installation is required — `torch.hub` handles the download automatically.

---

## Project Structure

```
nemo-playground/
├── live_stt.py          # Real-time STT pipeline (VAD + threading)
├── transcribe.py        # Batch file transcription utility
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── ROADMAP.md           # Planned features
└── TODO.md              # Outstanding tasks
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `No CUDA GPUs are available` | PyTorch CPU build installed | Reinstall with CUDA wheel (step 2) |
| `PortAudioError` / no microphone | PortAudio not found | Install PortAudio (step 4) |
| `torch.hub` download fails | Firewall / no internet | Pre-download the model or set `TORCH_HOME` |
| High CPU usage | Running on CPU | Use a CUDA-capable GPU |
| Clipped / truncated transcriptions | `--vad-threshold` too high | Lower to `0.3`–`0.4` |
| Too many false positives | `--vad-threshold` too low | Raise to `0.6`–`0.7` |

---

## License

See [LICENSE](LICENSE).
