# Whisper App — Streaming + Offline Redesign Spec
**Date:** 2026-06-11
**Builds on:** `whisper_app/app.py` (existing single-file app)
**Goal:** Replace the single toggle button with three separate Start/Stop/Transcribe buttons, add segment-by-segment streaming output, and enforce fully offline operation using only the cached mlx-community/whisper-large-v3-turbo model.

---

## Context

- Existing app: `whisper_app/app.py` — single file, one Record/Stop toggle, full transcript appears after recording stops
- Platform: macOS Apple Silicon (arm64), Python 3.11, 24 GB unified memory
- Model already cached: `mlx-community/whisper-large-v3-turbo` (~1.5 GB, in HF Hub local cache)
- No new dependencies — only what is already in `requirements.txt`

---

## Architecture

Single file `whisper_app/app.py`. Three targeted changes:

1. **`_ensure_offline()`** — module-level function, called once at startup; sets `HF_HUB_OFFLINE=1` if the model is already in cache
2. **`WhisperTranscriber`** — replaces `transcribe(wav_path)` with `transcribe_segments(audio, sample_rate)` generator
3. **`App`** — three buttons (Start / Stop / Transcribe) + five-state machine replacing the two-state toggle

---

## `_ensure_offline()`

Called once at the bottom of the module, before `App().mainloop()`.

```python
def _ensure_offline():
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    if cache.exists() and any(
        d.name.startswith("models--mlx-community--whisper-large-v3-turbo")
        for d in cache.iterdir()
        if d.is_dir()
    ):
        os.environ["HF_HUB_OFFLINE"] = "1"
```

- If the model cache exists → sets `HF_HUB_OFFLINE=1` so no network call is ever made
- If the model is not yet cached → runs normally (allows one-time download); subsequent runs will be offline
- `pathlib.Path` added to imports

---

## `WhisperTranscriber` — streaming generator

Old API (removed):
```python
def transcribe(self, wav_path: str) -> str
```

New API:
```python
def transcribe_segments(self, audio: np.ndarray, sample_rate: int = 16000) -> Generator[str, None, None]
```

- `audio`: float32 numpy array of the full recording (caller loads from WAV and passes the array)
- `sample_rate`: sample rate of the array (always 16000 in this app)
- Yields one `str` per non-empty 30-second chunk
- Chunk size: `30 * sample_rate` samples
- Each chunk is transcribed with `mlx_whisper.transcribe(chunk, path_or_hf_repo=self.MODEL)`
- Empty/whitespace results are skipped (not yielded)
- File cleanup moves to the caller — `WhisperTranscriber` no longer touches the filesystem

```python
CHUNK_SECONDS = 30

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

---

## State Machine

Five states; each defines which buttons are enabled and what colour they show.

| State | Start | Stop | Transcribe | Stored WAV |
|---|---|---|---|---|
| `idle` | ✅ green | ✗ disabled | ✗ disabled | none |
| `recording` | ✗ disabled | ✅ red | ✗ disabled | none |
| `stopped` | ✅ amber | ✗ disabled | ✅ blue | `_wav_path` set |
| `transcribing` | ✗ disabled | ✗ disabled | ✗ disabled | being consumed |
| `done` | ✅ green | ✗ disabled | ✗ disabled | deleted |

**Transitions:**
- `idle` → `recording`: user clicks Start → mic opens
- `recording` → `stopped`: user clicks Stop → mic closes, WAV saved to `_wav_path`
- `stopped` → `recording`: user clicks Start again → existing `_wav_path` deleted, new recording begins
- `stopped` → `transcribing`: user clicks Transcribe → background thread starts
- `transcribing` → `done`: all segments yielded → WAV deleted
- Any state → `idle` on error (via `_set_error`)

---

## GUI Layout

Three buttons in a horizontal row, replacing the single toggle. Below them: status label. Below that: scrollable text area (unchanged).

```
[ Start ]  [ Stop ]  [ Transcribe ]
          Recording…
┌─────────────────────────────────────┐
│ transcript text appears here…       │
│                                     │
└─────────────────────────────────────┘
```

---

## `_transcribe_async` (background thread)

```
load WAV as numpy array → run short-audio guard (< 0.5s) → delete WAV
→ iterate transcriber.transcribe_segments(audio, sr)
    → for each yielded text: self.after(0, _append_text, text)
→ self.after(0, _set_done)
```

If any exception: `self.after(0, _set_error, str(e))` → state returns to `idle`.

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| Mic permission denied | `_set_error("Microphone access denied")` → idle |
| Recording < 0.5 seconds | `_set_error("No audio captured")` → idle |
| Model not in cache (first run) | HF Hub downloads normally; no offline flag set yet |
| Transcription error | `_set_error(str(e))` → idle |
| Start pressed while `stopped` | Existing WAV deleted, state → recording |

---

## Tests to Update

`tests/test_transcriber.py` must be updated to match the new API:

- Remove `test_transcribe_returns_string` and `test_transcribe_deletes_wav`
- Add `test_transcribe_segments_yields_strings`: pass a 2-second silence array, collect all yielded values into a list, assert the generator completes without error and every item in the list is a `str` (list may be empty — silence is skipped)
- Add `test_transcribe_segments_chunks_long_audio`: pass a 65-second silence array (2+ chunks), collect yielded values, assert the generator completes without error (verifies chunking loop runs to completion without crashing)

`tests/test_recorder.py` — unchanged.

---

## Files Changed

| File | Change |
|---|---|
| `whisper_app/app.py` | Add `_ensure_offline`, rewrite `WhisperTranscriber`, rewrite `App` |
| `whisper_app/tests/test_transcriber.py` | Replace tests for new generator API |
| `whisper_app/requirements.txt` | No change |
| `whisper_app/tests/test_recorder.py` | No change |
