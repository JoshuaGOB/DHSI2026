# Whisper Test App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimal Tkinter app that records mic audio and transcribes it using `mlx-community/whisper-large-v3-turbo` to confirm the MLX inference stack works end-to-end on Apple Silicon macOS.

**Architecture:** Single `app.py` with three classes — `AudioRecorder` (sounddevice mic capture → temp WAV), `WhisperTranscriber` (mlx_whisper inference + temp file cleanup), and `App` (Tkinter GUI). Transcription runs in a background thread so the GUI stays responsive. TDD for the two non-GUI classes; manual verification for the GUI.

**Tech Stack:** Python 3.11, mlx-whisper, sounddevice, soundfile, numpy, tkinter (stdlib)

---

## File Structure

| File | Purpose |
|---|---|
| `whisper_app/requirements.txt` | Pinned dependencies |
| `whisper_app/app.py` | AudioRecorder + WhisperTranscriber + App (Tkinter) + entry point |
| `whisper_app/tests/__init__.py` | Makes tests a package |
| `whisper_app/tests/test_recorder.py` | Tests for AudioRecorder wav-writing |
| `whisper_app/tests/test_transcriber.py` | Tests for WhisperTranscriber with a generated silence WAV |

---

## Task 1: Dependencies

**Files:**
- Create: `whisper_app/requirements.txt`

- [ ] **Step 1: Create requirements.txt**

```
mlx-whisper
sounddevice
soundfile
numpy
```

- [ ] **Step 2: Install dependencies**

```bash
cd whisper_app
pip install -r requirements.txt
```

Expected: all packages install without error. `mlx` (~200 MB) is pulled in as a dependency of `mlx-whisper`.

- [ ] **Step 3: Verify imports**

```bash
python3 -c "import mlx_whisper; print('mlx_whisper OK')"
python3 -c "import sounddevice; print('sounddevice OK')"
python3 -c "import soundfile; print('soundfile OK')"
```

Expected:
```
mlx_whisper OK
sounddevice OK
soundfile OK
```

- [ ] **Step 4: Commit**

```bash
git add whisper_app/requirements.txt
git commit -m "feat: add whisper app requirements"
```

---

## Task 2: AudioRecorder

**Files:**
- Create: `whisper_app/app.py` (AudioRecorder only)
- Create: `whisper_app/tests/__init__.py`
- Create: `whisper_app/tests/test_recorder.py`

- [ ] **Step 1: Write the failing tests**

Create `whisper_app/tests/__init__.py` — empty file.

Create `whisper_app/tests/test_recorder.py`:

```python
import os
import sys
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app import AudioRecorder


def test_save_wav_produces_valid_file():
    recorder = AudioRecorder()
    recorder._buffer = [np.zeros((16000,), dtype=np.float32)]  # 1 second silence

    path = recorder._save_wav()

    assert os.path.exists(path)
    data, sr = sf.read(path)
    assert sr == 16000
    assert len(data) == 16000
    os.unlink(path)


def test_save_wav_concatenates_multiple_chunks():
    recorder = AudioRecorder()
    recorder._buffer = [
        np.zeros((8000,), dtype=np.float32),
        np.zeros((8000,), dtype=np.float32),
    ]

    path = recorder._save_wav()

    data, _ = sf.read(path)
    assert len(data) == 16000
    os.unlink(path)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd whisper_app
python3 -m pytest tests/test_recorder.py -v
```

Expected: `ImportError: cannot import name 'AudioRecorder' from 'app'`

- [ ] **Step 3: Create app.py with AudioRecorder**

Create `whisper_app/app.py`:

```python
import os
import threading
import tempfile

import numpy as np
import sounddevice as sd
import soundfile as sf


class AudioRecorder:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._buffer: list[np.ndarray] = []
        self._stream = None

    def start(self):
        self._buffer = []
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> str:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        return self._save_wav()

    def _callback(self, indata, frames, time, status):
        self._buffer.append(indata.copy().flatten())

    def _save_wav(self) -> str:
        audio = np.concatenate(self._buffer) if self._buffer else np.zeros((1,), dtype=np.float32)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, audio, self.sample_rate)
        return tmp.name
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_recorder.py -v
```

Expected:
```
PASSED tests/test_recorder.py::test_save_wav_produces_valid_file
PASSED tests/test_recorder.py::test_save_wav_concatenates_multiple_chunks
```

- [ ] **Step 5: Commit**

```bash
git add whisper_app/app.py whisper_app/tests/__init__.py whisper_app/tests/test_recorder.py
git commit -m "feat: add AudioRecorder with wav-writing tests"
```

---

## Task 3: WhisperTranscriber

**Files:**
- Modify: `whisper_app/app.py` (add import + WhisperTranscriber class)
- Create: `whisper_app/tests/test_transcriber.py`

> **Note:** Running the tests in this task will download `mlx-community/whisper-large-v3-turbo` (~1.5 GB) from HF Hub on first run. Subsequent runs use the local cache. Expect 5–15 minutes on first run depending on connection speed.

- [ ] **Step 1: Write the failing tests**

Create `whisper_app/tests/test_transcriber.py`:

```python
import os
import sys
import numpy as np
import soundfile as sf
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app import WhisperTranscriber


def _make_silence_wav(duration_sec: float = 2.0, sample_rate: int = 16000) -> str:
    audio = np.zeros(int(duration_sec * sample_rate), dtype=np.float32)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sample_rate)
    return tmp.name


def test_transcribe_returns_string():
    wav_path = _make_silence_wav()
    transcriber = WhisperTranscriber()
    result = transcriber.transcribe(wav_path)
    # wav_path already deleted by transcribe(); don't unlink again
    assert isinstance(result, str), f"Expected str, got {type(result)}"


def test_transcribe_deletes_wav():
    wav_path = _make_silence_wav()
    transcriber = WhisperTranscriber()
    transcriber.transcribe(wav_path)
    assert not os.path.exists(wav_path), "Temp WAV was not cleaned up after transcription"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_transcriber.py -v
```

Expected: `ImportError: cannot import name 'WhisperTranscriber' from 'app'`

- [ ] **Step 3: Add WhisperTranscriber to app.py**

Add `import mlx_whisper` to the imports block at the top of `whisper_app/app.py` (after the existing imports), then append this class after `AudioRecorder`:

```python
import mlx_whisper  # add to imports block at top


class WhisperTranscriber:
    MODEL = "mlx-community/whisper-large-v3-turbo"

    def transcribe(self, wav_path: str) -> str:
        try:
            result = mlx_whisper.transcribe(wav_path, path_or_hf_repo=self.MODEL)
            return result.get("text", "").strip()
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python3 -m pytest tests/test_transcriber.py -v
```

First run: HF Hub download progress will print to the terminal (~1.5 GB). After download:

```
PASSED tests/test_transcriber.py::test_transcribe_returns_string
PASSED tests/test_transcriber.py::test_transcribe_deletes_wav
```

- [ ] **Step 5: Run all tests to confirm nothing regressed**

```bash
python3 -m pytest tests/ -v
```

Expected: all 4 tests pass.

- [ ] **Step 6: Commit**

```bash
git add whisper_app/app.py whisper_app/tests/test_transcriber.py
git commit -m "feat: add WhisperTranscriber with mlx-whisper backend"
```

---

## Task 4: Tkinter GUI

**Files:**
- Modify: `whisper_app/app.py` (add imports + App class + `__main__` entry point)

No automated tests for the GUI — verified manually in Task 5.

- [ ] **Step 1: Add App class and entry point to app.py**

Add these imports to the imports block at the top of `whisper_app/app.py`:

```python
import tkinter as tk
from tkinter import scrolledtext
```

Then append the `App` class and entry point after `WhisperTranscriber`:

```python
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Whisper Test")
        self.resizable(False, False)

        self._recorder = AudioRecorder()
        self._transcriber = WhisperTranscriber()
        self._recording = False

        self._btn = tk.Button(
            self, text="Record", width=20, command=self._toggle,
            bg="#4CAF50", fg="white", font=("Helvetica", 14),
        )
        self._btn.pack(pady=(16, 8), padx=24)

        self._status = tk.Label(self, text="Idle", font=("Helvetica", 11), fg="gray")
        self._status.pack(pady=(0, 8))

        self._text = scrolledtext.ScrolledText(
            self, width=60, height=12, wrap=tk.WORD,
            state=tk.DISABLED, font=("Helvetica", 12),
        )
        self._text.pack(padx=16, pady=(0, 16))

    def _toggle(self):
        if not self._recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        self._recording = True
        self._btn.config(text="Stop", bg="#f44336")
        self._status.config(text="Recording…", fg="red")
        try:
            self._recorder.start()
        except Exception as e:
            self._set_error(str(e))

    def _stop_recording(self):
        self._recording = False
        self._btn.config(state=tk.DISABLED)
        self._status.config(text="Transcribing…", fg="orange")
        threading.Thread(target=self._transcribe_async, daemon=True).start()

    def _transcribe_async(self):
        try:
            wav_path = self._recorder.stop()
            # Guard: reject recordings shorter than 0.5 seconds
            data, sr = sf.read(wav_path)
            if len(data) / sr < 0.5:
                os.unlink(wav_path)
                self.after(0, self._set_error, "No audio captured")
                return
            text = self._transcriber.transcribe(wav_path)
            self.after(0, self._append_text, text)
            self.after(0, self._set_idle)
        except Exception as e:
            self.after(0, self._set_error, str(e))

    def _append_text(self, text: str):
        self._text.config(state=tk.NORMAL)
        if text:
            self._text.insert(tk.END, text + "\n")
        self._text.config(state=tk.DISABLED)
        self._text.see(tk.END)

    def _set_idle(self):
        self._btn.config(text="Record", bg="#4CAF50", state=tk.NORMAL)
        self._status.config(text="Done", fg="green")

    def _set_error(self, msg: str):
        self._btn.config(text="Record", bg="#4CAF50", state=tk.NORMAL)
        self._status.config(text=f"Error: {msg}", fg="red")
        self._recording = False


if __name__ == "__main__":
    App().mainloop()
```

- [ ] **Step 2: Run all tests to confirm nothing broke**

```bash
python3 -m pytest tests/ -v
```

Expected: all 4 tests pass.

- [ ] **Step 3: Commit**

```bash
git add whisper_app/app.py
git commit -m "feat: add Tkinter GUI with record/stop/transcribe flow"
```

---

## Task 5: Manual end-to-end verification

**Files:** none (verification only)

- [ ] **Step 1: Launch the app**

```bash
cd whisper_app
python3 app.py
```

Expected: a small window appears with a green "Record" button, "Idle" status, and an empty text area.

- [ ] **Step 2: Test recording**

Click **Record**. Expected:
- Button turns red and shows "Stop"
- Status label shows "Recording…"
- macOS may display a microphone permission prompt — grant it in System Settings → Privacy & Security → Microphone

- [ ] **Step 3: Test transcription**

Speak a sentence (e.g. in Spanish: *"Hola, esto es una prueba de transcripción."*), then click **Stop**. Expected:
- Status shows "Transcribing…" (3–10 seconds)
- Transcript appears in the text area
- Status shows "Done" in green
- Button returns to green "Record"

- [ ] **Step 4: Test a second cycle**

Click **Record** again, speak another sentence, click **Stop**. Verify the new transcript appends below the first without errors or crashes.

- [ ] **Step 5: Final commit**

```bash
git add whisper_app/
git commit -m "feat: complete whisper test app — mlx-large-v3-turbo verified"
```
