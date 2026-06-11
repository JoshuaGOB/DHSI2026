# Whisper Streaming + Offline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `whisper_app/app.py` to run fully offline, stream transcription segment-by-segment, and expose three separate Start / Stop / Transcribe buttons.

**Architecture:** Three targeted changes to the existing single file: (1) add `_ensure_offline()` that sets `HF_HUB_OFFLINE=1` when the model cache exists; (2) replace `WhisperTranscriber.transcribe(wav_path)` with a `transcribe_segments(audio, sample_rate)` generator that slices the numpy array into 30-second chunks and yields text per chunk; (3) rewrite `App` with three buttons and a five-state machine (`idle → recording → stopped → transcribing → done`).

**Tech Stack:** Python 3.11, mlx-whisper 0.4.3, sounddevice, soundfile, numpy, tkinter (stdlib), pathlib (stdlib)

---

## File Structure

| File | Change |
|---|---|
| `whisper_app/app.py` | Add `_ensure_offline`, rewrite `WhisperTranscriber`, rewrite `App` |
| `whisper_app/tests/test_offline.py` | New — tests for `_ensure_offline` |
| `whisper_app/tests/test_transcriber.py` | Replace — update for new generator API |
| `whisper_app/tests/test_recorder.py` | No change |

---

## Task 1: `_ensure_offline()`

**Files:**
- Modify: `whisper_app/app.py` (add import + function)
- Create: `whisper_app/tests/test_offline.py`

- [ ] **Step 1: Write the failing tests**

Create `whisper_app/tests/test_offline.py`:

```python
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app import _ensure_offline


def test_sets_env_when_model_dir_exists(tmp_path):
    hub_dir = tmp_path / "hub"
    hub_dir.mkdir()
    (hub_dir / "models--mlx-community--whisper-large-v3-turbo").mkdir()

    prev = os.environ.pop("HF_HUB_OFFLINE", None)
    try:
        _ensure_offline(cache_dir=hub_dir)
        assert os.environ.get("HF_HUB_OFFLINE") == "1"
    finally:
        os.environ.pop("HF_HUB_OFFLINE", None)
        if prev is not None:
            os.environ["HF_HUB_OFFLINE"] = prev


def test_does_not_set_env_when_cache_missing(tmp_path):
    prev = os.environ.pop("HF_HUB_OFFLINE", None)
    try:
        _ensure_offline(cache_dir=tmp_path / "nonexistent")
        assert "HF_HUB_OFFLINE" not in os.environ
    finally:
        os.environ.pop("HF_HUB_OFFLINE", None)
        if prev is not None:
            os.environ["HF_HUB_OFFLINE"] = prev
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/joga/Downloads/DHSI2026/huggingface_demo/whisper_app
/Users/joga/miniconda3/bin/python3 -m pytest tests/test_offline.py -v
```

Expected: `ImportError: cannot import name '_ensure_offline' from 'app'`

- [ ] **Step 3: Add `_ensure_offline` to app.py**

Add `from pathlib import Path` to the imports block at the top of `whisper_app/app.py` (after `import os`).

Add this function immediately after the imports block, before `class AudioRecorder`:

```python
def _ensure_offline(cache_dir: Path = None) -> None:
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if cache_dir.exists() and any(
        d.name.startswith("models--mlx-community--whisper-large-v3-turbo")
        for d in cache_dir.iterdir()
        if d.is_dir()
    ):
        os.environ["HF_HUB_OFFLINE"] = "1"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/Users/joga/miniconda3/bin/python3 -m pytest tests/test_offline.py -v
```

Expected:
```
PASSED tests/test_offline.py::test_sets_env_when_model_dir_exists
PASSED tests/test_offline.py::test_does_not_set_env_when_cache_missing
```

- [ ] **Step 5: Run full test suite**

```bash
/Users/joga/miniconda3/bin/python3 -m pytest tests/ -v
```

Expected: all 6 tests pass (2 recorder + 2 transcriber + 2 offline).

- [ ] **Step 6: Commit**

```bash
git -C /Users/joga/Downloads/DHSI2026/huggingface_demo add whisper_app/app.py whisper_app/tests/test_offline.py
git -C /Users/joga/Downloads/DHSI2026/huggingface_demo commit -m "feat: add _ensure_offline with HF hub cache detection"
```

---

## Task 2: `WhisperTranscriber` — streaming generator

**Files:**
- Modify: `whisper_app/app.py` (replace `transcribe` with `transcribe_segments`, add `CHUNK_SECONDS`)
- Modify: `whisper_app/tests/test_transcriber.py` (replace all tests for new API)

- [ ] **Step 1: Replace test_transcriber.py with new tests**

Overwrite `whisper_app/tests/test_transcriber.py` with:

```python
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app import WhisperTranscriber


def test_transcribe_segments_yields_strings():
    audio = np.zeros(int(2.0 * 16000), dtype=np.float32)
    transcriber = WhisperTranscriber()
    results = list(transcriber.transcribe_segments(audio, sample_rate=16000))
    # silence may yield nothing; every yielded item must be a str
    assert all(isinstance(r, str) for r in results)


def test_transcribe_segments_chunks_long_audio():
    # 65 seconds — forces 3 chunks (0-30s, 30-60s, 60-65s)
    audio = np.zeros(int(65.0 * 16000), dtype=np.float32)
    transcriber = WhisperTranscriber()
    # must complete without error; silence may yield nothing
    results = list(transcriber.transcribe_segments(audio, sample_rate=16000))
    assert all(isinstance(r, str) for r in results)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/joga/Downloads/DHSI2026/huggingface_demo/whisper_app
/Users/joga/miniconda3/bin/python3 -m pytest tests/test_transcriber.py -v
```

Expected: `AttributeError: 'WhisperTranscriber' object has no attribute 'transcribe_segments'`

- [ ] **Step 3: Update WhisperTranscriber in app.py**

Add `CHUNK_SECONDS = 30` as a module-level constant immediately after the `_ensure_offline` function.

Replace the entire `WhisperTranscriber` class with:

```python
class WhisperTranscriber:
    MODEL = "mlx-community/whisper-large-v3-turbo"

    def transcribe_segments(self, audio: np.ndarray, sample_rate: int = 16000):
        chunk_size = CHUNK_SECONDS * sample_rate
        for start in range(0, len(audio), chunk_size):
            chunk = audio[start : start + chunk_size].astype(np.float32)
            result = mlx_whisper.transcribe(chunk, path_or_hf_repo=self.MODEL)
            text = result.get("text", "").strip()
            if text:
                yield text
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/Users/joga/miniconda3/bin/python3 -m pytest tests/test_transcriber.py -v
```

Expected (model already cached, runs quickly):
```
PASSED tests/test_transcriber.py::test_transcribe_segments_yields_strings
PASSED tests/test_transcriber.py::test_transcribe_segments_chunks_long_audio
```

- [ ] **Step 5: Run full test suite**

```bash
/Users/joga/miniconda3/bin/python3 -m pytest tests/ -v
```

Expected: all 6 tests pass.

- [ ] **Step 6: Commit**

```bash
git -C /Users/joga/Downloads/DHSI2026/huggingface_demo add whisper_app/app.py whisper_app/tests/test_transcriber.py
git -C /Users/joga/Downloads/DHSI2026/huggingface_demo commit -m "feat: replace transcribe with transcribe_segments streaming generator"
```

---

## Task 3: Three-button App with state machine

**Files:**
- Modify: `whisper_app/app.py` (replace entire `App` class and `__main__` block)

No automated tests for the GUI — verified manually in Task 4.

- [ ] **Step 1: Replace the App class and entry point in app.py**

Replace everything from `class App(tk.Tk):` through the end of the file with:

```python
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Whisper")
        self.resizable(False, False)

        self._recorder = AudioRecorder()
        self._transcriber = WhisperTranscriber()
        self._wav_path: str | None = None

        btn_row = tk.Frame(self)
        btn_row.pack(pady=(16, 8), padx=24)

        self._btn_start = tk.Button(
            btn_row, text="Start", width=10, command=self._on_start,
            fg="white", font=("Helvetica", 13),
        )
        self._btn_start.pack(side=tk.LEFT, padx=4)

        self._btn_stop = tk.Button(
            btn_row, text="Stop", width=10, command=self._on_stop,
            fg="white", font=("Helvetica", 13),
        )
        self._btn_stop.pack(side=tk.LEFT, padx=4)

        self._btn_transcribe = tk.Button(
            btn_row, text="Transcribe", width=10, command=self._on_transcribe,
            fg="white", font=("Helvetica", 13),
        )
        self._btn_transcribe.pack(side=tk.LEFT, padx=4)

        self._status = tk.Label(self, text="Ready", font=("Helvetica", 11), fg="gray")
        self._status.pack(pady=(0, 8))

        self._text = scrolledtext.ScrolledText(
            self, width=60, height=12, wrap=tk.WORD,
            state=tk.DISABLED, font=("Helvetica", 12),
        )
        self._text.pack(padx=16, pady=(0, 16))

        self._set_state("idle")

    # ── button handlers ───────────────────────────────────────────────────────

    def _on_start(self):
        if self._wav_path and os.path.exists(self._wav_path):
            os.unlink(self._wav_path)
            self._wav_path = None
        self._set_state("recording")
        try:
            self._recorder.start()
        except Exception as e:
            self._set_error(str(e))

    def _on_stop(self):
        self._set_state("idle")  # disable all buttons while WAV is being written
        try:
            self._wav_path = self._recorder.stop()
            self._set_state("stopped")
        except Exception as e:
            self._set_error(str(e))

    def _on_transcribe(self):
        self._set_state("transcribing")
        threading.Thread(target=self._transcribe_async, daemon=True).start()

    # ── background transcription ──────────────────────────────────────────────

    def _transcribe_async(self):
        wav_path, self._wav_path = self._wav_path, None
        try:
            audio, sr = sf.read(wav_path)
            os.unlink(wav_path)
            wav_path = None  # marked cleaned up
            if len(audio) / sr < 0.5:
                self.after(0, self._set_error, "No audio captured")
                return
            for text in self._transcriber.transcribe_segments(audio.astype(np.float32), sr):
                self.after(0, self._append_text, text)
            self.after(0, self._set_state, "done")
        except Exception as e:
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)
            self.after(0, self._set_error, str(e))

    # ── GUI helpers ───────────────────────────────────────────────────────────

    _STATES = {
        #             start              stop               transcribe          status label
        "idle":        [(True,  "#4CAF50"), (False, "#9E9E9E"), (False, "#9E9E9E"), ("Ready",                        "gray")],
        "recording":   [(False, "#9E9E9E"), (True,  "#f44336"), (False, "#9E9E9E"), ("Recording…",              "red")],
        "stopped":     [(True,  "#FF9800"), (False, "#9E9E9E"), (True,  "#2196F3"), ("Audio captured — press Transcribe", "#2196F3")],
        "transcribing":[(False, "#9E9E9E"), (False, "#9E9E9E"), (False, "#9E9E9E"), ("Transcribing…",           "orange")],
        "done":        [(True,  "#4CAF50"), (False, "#9E9E9E"), (False, "#9E9E9E"), ("Done",                         "green")],
    }

    def _set_state(self, state: str):
        s_start, s_stop, s_trans, (label, fg) = self._STATES[state]
        for btn, (enabled, bg) in zip(
            [self._btn_start, self._btn_stop, self._btn_transcribe],
            [s_start, s_stop, s_trans],
        ):
            btn.config(state=tk.NORMAL if enabled else tk.DISABLED, bg=bg)
        self._status.config(text=label, fg=fg)

    def _append_text(self, text: str):
        self._text.config(state=tk.NORMAL)
        if text:
            self._text.insert(tk.END, text + "\n")
        self._text.config(state=tk.DISABLED)
        self._text.see(tk.END)

    def _set_error(self, msg: str):
        self._status.config(text=f"Error: {msg}", fg="red")
        self._set_state("idle")


if __name__ == "__main__":
    _ensure_offline()
    App().mainloop()
```

- [ ] **Step 2: Run full test suite to confirm nothing broke**

```bash
cd /Users/joga/Downloads/DHSI2026/huggingface_demo/whisper_app
/Users/joga/miniconda3/bin/python3 -m pytest tests/ -v
```

Expected: all 6 tests pass.

- [ ] **Step 3: Commit**

```bash
git -C /Users/joga/Downloads/DHSI2026/huggingface_demo add whisper_app/app.py
git -C /Users/joga/Downloads/DHSI2026/huggingface_demo commit -m "feat: add three-button UI with five-state machine and segment streaming"
```

---

## Task 4: Manual end-to-end verification

**Files:** none (verification only)

- [ ] **Step 1: Launch the app**

```bash
cd /Users/joga/Downloads/DHSI2026/huggingface_demo/whisper_app
python3 app.py
```

Expected: window opens with three buttons in a row — **Start** (green), **Stop** (grey/disabled), **Transcribe** (grey/disabled). Status shows "Ready".

- [ ] **Step 2: Verify Start → Stop flow**

Click **Start**. Expected:
- Start goes grey/disabled
- Stop turns red/enabled
- Transcribe stays grey/disabled
- Status shows "Recording…"

Speak a sentence, then click **Stop**. Expected:
- Stop goes grey/disabled
- Start turns amber/enabled
- Transcribe turns blue/enabled
- Status shows "Audio captured — press Transcribe"

- [ ] **Step 3: Verify streaming transcription**

Click **Transcribe**. Expected:
- All three buttons disable
- Status shows "Transcribing…"
- Transcript text appears segment by segment in the text area (for recordings < 30s, one burst; for longer, multiple bursts)
- Status shows "Done" in green
- Start re-enables (green); Stop and Transcribe stay disabled

- [ ] **Step 4: Verify re-record flow**

Click **Start** again. Expected: new recording begins (previous audio discarded). Record briefly, click **Stop**, click **Transcribe**. Verify new transcript appends below the first.

- [ ] **Step 5: Verify offline mode**

Check the terminal for any network activity or HF Hub warnings — there should be none. Confirm the app works with wifi disabled (if convenient).

- [ ] **Step 6: Final commit**

```bash
git -C /Users/joga/Downloads/DHSI2026/huggingface_demo add whisper_app/
git -C /Users/joga/Downloads/DHSI2026/huggingface_demo commit -m "feat: complete streaming offline whisper app — verified"
```
