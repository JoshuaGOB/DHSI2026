# Whisper App — Live Transcription + Clear/Save Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace post-recording Transcribe button with live transcription that streams results as audio is captured. Also adds Clear and Save File buttons below the transcript area.

**Architecture:** Two specs, one file. `AudioRecorder.stop()` simplified to stream-close only; new `_live_loop()` daemon thread polls the audio buffer every 0.5 s and transcribes complete 10-second chunks as they accumulate; `_on_stop()` closes the stream and signals the live loop to process the tail; `App.__init__` drops the Transcribe button and adds Clear + Save File below the ScrolledText. State machine simplified from 5 states to 4 (idle / recording / finalizing / done).

**Tech Stack:** Python 3.11, mlx-whisper, sounddevice, soundfile, numpy, tkinter (stdlib), threading (stdlib)

**Specs:**
- `docs/superpowers/specs/2026-06-11-whisper-live-transcription-design.md`
- `docs/superpowers/specs/2026-06-11-whisper-clear-save-design.md`

---

## File Structure

| File | Change |
|---|---|
| `whisper_app/app.py` | All changes — see tasks below |
| `whisper_app/tests/` | No changes |

---

## Task 1: AudioRecorder.stop() + LIVE_CHUNK_SECONDS

**Files:**
- Modify: `whisper_app/app.py`

- [ ] **Step 1: Add LIVE_CHUNK_SECONDS constant**

In `app.py`, after `CHUNK_SECONDS = 30`, add:

```python
LIVE_CHUNK_SECONDS = 10
```

- [ ] **Step 2: Simplify AudioRecorder.stop()**

Replace the existing `stop()` method:

```python
def stop(self) -> str:
    if self._stream:
        self._stream.stop()
        self._stream.close()
        self._stream = None
    return self._save_wav()
```

with:

```python
def stop(self) -> None:
    if self._stream:
        self._stream.stop()
        self._stream.close()
        self._stream = None
```

`_save_wav()` stays in the class — `test_recorder.py` calls it directly.

- [ ] **Step 3: Run full test suite**

```bash
/Users/joga/miniconda3/bin/python3 -m pytest /Users/joga/Downloads/DHSI2026/huggingface_demo/whisper_app/tests/ -v
```

Expected: all 6 tests pass (2 recorder + 2 transcriber + 2 offline). The recorder tests call `_save_wav()` directly and are unaffected by the `stop()` change.

- [ ] **Step 4: Commit**

```bash
git -C /Users/joga/Downloads/DHSI2026/huggingface_demo add whisper_app/app.py
git -C /Users/joga/Downloads/DHSI2026/huggingface_demo commit -m "refactor: simplify AudioRecorder.stop(), add LIVE_CHUNK_SECONDS"
```

---

## Task 2: Rewrite App class — live transcription

**Files:**
- Modify: `whisper_app/app.py`

This task rewrites the `App` class to remove the Transcribe button and replace post-recording transcription with a live background loop.

- [ ] **Step 1: Update imports in App.__init__**

Remove `self._wav_path` from `App.__init__`. The full new `__init__` (up to `self._set_state`):

```python
def __init__(self):
    super().__init__()
    self.title("Whisper")
    self.resizable(False, False)

    self._recorder = AudioRecorder()
    self._transcriber = WhisperTranscriber()
    self._live_stop: threading.Event | None = None

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

    self._status = tk.Label(self, text="Ready", font=("Helvetica", 11), fg="gray")
    self._status.pack(pady=(0, 8))

    self._text = scrolledtext.ScrolledText(
        self, width=60, height=12, wrap=tk.WORD,
        state=tk.DISABLED, font=("Helvetica", 12),
    )
    self._text.pack(padx=16, pady=(0, 16))

    self._set_state("idle")
```

- [ ] **Step 2: Replace _STATES and _set_state**

Replace the existing `_STATES` dict and `_set_state` method with:

```python
_STATES = {
    #              start               stop                status label
    "idle":       [(True,  "#4CAF50"), (False, "#9E9E9E"), ("Ready",      "gray")],
    "recording":  [(False, "#9E9E9E"), (True,  "#f44336"), ("Recording…", "red")],
    "finalizing": [(False, "#9E9E9E"), (False, "#9E9E9E"), ("Finishing…", "orange")],
    "done":       [(True,  "#4CAF50"), (False, "#9E9E9E"), ("Done",       "green")],
}

def _set_state(self, state: str):
    s_start, s_stop, (label, fg) = self._STATES[state]
    for btn, (enabled, bg) in zip(
        [self._btn_start, self._btn_stop],
        [s_start, s_stop],
    ):
        btn.config(state=tk.NORMAL if enabled else tk.DISABLED, bg=bg)
    self._status.config(text=label, fg=fg)
```

- [ ] **Step 3: Replace _on_start**

Replace the existing `_on_start` with:

```python
def _on_start(self):
    self._set_state("recording")
    self._live_stop = threading.Event()
    try:
        self._recorder.start()
        threading.Thread(target=self._live_loop, args=(self._live_stop,), daemon=True).start()
    except Exception as e:
        self._set_error(str(e))
```

- [ ] **Step 4: Replace _on_stop**

Replace the existing `_on_stop` with:

```python
def _on_stop(self):
    self._set_state("finalizing")
    self._recorder.stop()
    self._live_stop.set()
```

- [ ] **Step 5: Add _live_loop**

Add this method to the App class (after `_on_stop`, before `_set_state`):

```python
def _live_loop(self, stop_event: threading.Event):
    chunk_samples = LIVE_CHUNK_SECONDS * self._recorder.sample_rate
    processed = 0
    while not stop_event.is_set():
        with self._recorder._lock:
            all_audio = (
                np.concatenate(self._recorder._buffer)
                if self._recorder._buffer
                else np.zeros(0, dtype=np.float32)
            )
        if len(all_audio) - processed >= chunk_samples:
            chunk = all_audio[processed : processed + chunk_samples].astype(np.float32)
            processed += chunk_samples
            for text in self._transcriber.transcribe_segments(chunk, self._recorder.sample_rate):
                self.after(0, self._append_text, text)
        else:
            stop_event.wait(timeout=0.5)
    # Transcribe tail after stop
    with self._recorder._lock:
        all_audio = (
            np.concatenate(self._recorder._buffer)
            if self._recorder._buffer
            else np.zeros(0, dtype=np.float32)
        )
    remaining = all_audio[processed:].astype(np.float32)
    if len(remaining) / self._recorder.sample_rate >= 0.5:
        for text in self._transcriber.transcribe_segments(remaining, self._recorder.sample_rate):
            self.after(0, self._append_text, text)
    self.after(0, self._set_state, "done")
```

- [ ] **Step 6: Remove _on_transcribe and _transcribe_async**

Delete the following methods entirely from the App class:

```python
def _on_transcribe(self):
    ...

def _transcribe_async(self):
    ...
```

- [ ] **Step 7: Run full test suite**

```bash
/Users/joga/miniconda3/bin/python3 -m pytest /Users/joga/Downloads/DHSI2026/huggingface_demo/whisper_app/tests/ -v
```

Expected: all 6 tests pass.

- [ ] **Step 8: Commit**

```bash
git -C /Users/joga/Downloads/DHSI2026/huggingface_demo add whisper_app/app.py
git -C /Users/joga/Downloads/DHSI2026/huggingface_demo commit -m "feat: live transcription — replace Transcribe button with offset-pointer live loop"
```

---

## Task 3: Clear + Save File buttons

**Files:**
- Modify: `whisper_app/app.py`

Implements `docs/superpowers/specs/2026-06-11-whisper-clear-save-design.md`.

- [ ] **Step 1: Add filedialog import**

In the imports block at the top of `app.py`, change:

```python
from tkinter import scrolledtext
```

to:

```python
from tkinter import scrolledtext, filedialog
```

- [ ] **Step 2: Add action_row frame and buttons in __init__**

After the line `self._text.pack(padx=16, pady=(0, 16))` and before `self._set_state("idle")`, add:

```python
action_row = tk.Frame(self)
action_row.pack(pady=(0, 16))

self._btn_clear = tk.Button(
    action_row, text="Clear", width=10, command=self._on_clear,
    bg="#9E9E9E", fg="white", font=("Helvetica", 13),
)
self._btn_clear.pack(side=tk.LEFT, padx=4)

self._btn_save = tk.Button(
    action_row, text="Save File", width=10, command=self._on_save,
    bg="#2196F3", fg="white", font=("Helvetica", 13),
)
self._btn_save.pack(side=tk.LEFT, padx=4)
```

- [ ] **Step 3: Add _on_clear method**

Add after `_set_state`:

```python
def _on_clear(self):
    self._text.config(state=tk.NORMAL)
    self._text.delete("1.0", tk.END)
    self._text.config(state=tk.DISABLED)
```

- [ ] **Step 4: Add _on_save method**

Add after `_on_clear`:

```python
def _on_save(self):
    path = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
    )
    if not path:
        return
    content = self._text.get("1.0", tk.END)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
```

- [ ] **Step 5: Run full test suite**

```bash
/Users/joga/miniconda3/bin/python3 -m pytest /Users/joga/Downloads/DHSI2026/huggingface_demo/whisper_app/tests/ -v
```

Expected: all 6 tests pass.

- [ ] **Step 6: Commit**

```bash
git -C /Users/joga/Downloads/DHSI2026/huggingface_demo add whisper_app/app.py
git -C /Users/joga/Downloads/DHSI2026/huggingface_demo commit -m "feat: add Clear and Save File buttons"
```

---

## Task 4: Manual end-to-end verification

**Files:** none (verification only)

- [ ] **Step 1: Launch the app**

```bash
/Users/joga/miniconda3/bin/python3 /Users/joga/Downloads/DHSI2026/huggingface_demo/whisper_app/app.py
```

Expected: window with **[Start] [Stop]** in the top row, status "Ready", empty transcript area, **[Clear] [Save File]** below.

- [ ] **Step 2: Verify live transcription flow**

Click **Start**. Expected:
- Start goes grey/disabled
- Stop turns red/enabled
- Status shows "Recording…"

Speak for 10+ seconds. Expected: text begins appearing in the transcript area before you stop.

Click **Stop**. Expected:
- Both buttons immediately disable
- Status shows "Finishing…"
- Any remaining words appear in the text area
- Status shows "Done" in green
- Start re-enables

- [ ] **Step 3: Verify Clear button**

Click **Clear**. Expected: text area empties. Start and Stop remain in their current state.

- [ ] **Step 4: Verify Save File button**

Click **Start**, speak briefly, click **Stop** (wait for Done). Click **Save File**. Expected:
- macOS save dialog appears
- Choose a location and save as `.txt`
- Verify file content matches the transcript

- [ ] **Step 5: Verify re-record**

Click **Start** again (from Done state). Record new audio. Expected: new text appends below any existing content.

- [ ] **Step 6: Final commit**

```bash
git -C /Users/joga/Downloads/DHSI2026/huggingface_demo add whisper_app/
git -C /Users/joga/Downloads/DHSI2026/huggingface_demo commit -m "feat: complete live transcription with Clear and Save — verified"
```
