# Whisper Test App — Design Spec
**Date:** 2026-06-11
**Goal:** Minimal Tkinter app to confirm `mlx-community/whisper-large-v3-turbo` works end-to-end for live microphone transcription on macOS Apple Silicon.

---

## Context

- **Platform:** macOS, Apple Silicon (arm64), 24 GB unified memory
- **Python:** 3.11
- **Primary language:** Spanish / French / Portuguese (multilingual model required)
- **Inference stack:** MLX framework (`mlx-whisper`) — chosen to validate the path that will be used in the real word processor app

---

## Architecture

Single-file application: `whisper_app/app.py` plus `whisper_app/requirements.txt`. No submodules or packages. The model is downloaded from HF Hub on first run (~1.5 GB) and cached locally for all subsequent runs.

---

## Components

### 1. `AudioRecorder`
Captures microphone input using `sounddevice` in a background thread.

- Records at 16 kHz, mono (the sample rate Whisper expects)
- Accumulates audio frames into a NumPy buffer
- On stop, writes the buffer to a temporary `.wav` file via `soundfile`
- Exposes `start()`, `stop() -> wav_path` interface

### 2. `WhisperTranscriber`
Wraps `mlx_whisper.transcribe()`.

- Accepts a `.wav` file path
- Returns the transcript string from `result["text"]`
- Model (`mlx-community/whisper-large-v3-turbo`) is loaded lazily on first call
- Cleans up the temp `.wav` file after transcription regardless of success/failure

### 3. `App` (Tkinter GUI)
A small, fixed-size window containing:

- **Record / Stop button** — toggles between recording and idle states
- **Status label** — displays current state: `Idle`, `Recording…`, `Transcribing…`, `Done`, or an error message
- **Scrollable text area** — each completed transcript is appended on a new line; read-only

Transcription runs in a background thread so the GUI stays responsive during model inference.

---

## Data Flow

```
[User clicks Record]
    → AudioRecorder.start()          # sounddevice begins capturing 16kHz mono
[User clicks Stop]
    → AudioRecorder.stop()           # writes buffer to temp .wav
    → WhisperTranscriber.transcribe(wav_path)   # mlx_whisper.transcribe()
    → result["text"]
    → append to Tkinter text area
    → delete temp .wav
```

---

## Dependencies (`requirements.txt`)

```
mlx-whisper
sounddevice
soundfile
numpy
```

> `tkinter` ships with macOS Python — no install needed.

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| Mic permission denied | Status label: "Microphone access denied — check System Settings → Privacy" |
| Model not yet downloaded | Status label: "Downloading model (first run ~1.5 GB)…" — GUI remains responsive |
| Transcription error | Status label shows exception message; temp file always cleaned up |
| No audio recorded (zero-length) | Guard check before calling transcribe; status label: "No audio captured" |

---

## File Layout

```
whisper_app/
├── app.py            # entire application
└── requirements.txt  # mlx-whisper, sounddevice, soundfile, numpy
```

---

## Success Criteria

1. `pip install -r requirements.txt` completes without error on Apple Silicon
2. App window opens
3. Clicking Record captures mic audio (status shows "Recording…")
4. Clicking Stop triggers transcription (status shows "Transcribing…")
5. Correct transcript appears in the text area within a few seconds
6. App handles errors gracefully without crashing
