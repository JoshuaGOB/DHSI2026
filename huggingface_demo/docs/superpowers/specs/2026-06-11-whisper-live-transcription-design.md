# Whisper App — Live Transcription Design
**Date:** 2026-06-11
**Builds on:** `whisper_app/app.py` (three-button streaming app)
**Goal:** Replace post-recording transcription with live transcription that starts immediately when recording begins. Stop finalizes any remaining audio. The Transcribe button is removed.

Lands in the same implementation plan as `2026-06-11-whisper-clear-save-design.md` (Clear + Save File buttons).

---

## Architecture

### Approach: Offset-pointer reader

A `_live_loop(stop_event)` daemon thread holds an integer `processed` tracking how many samples have already been transcribed. It polls `_recorder._buffer` under the existing lock every 0.5 seconds. When at least `LIVE_CHUNK_SECONDS * sample_rate` new samples have accumulated past the offset, it slices that chunk and transcribes it. When `stop_event` is set (by `_on_stop`), the loop exits its poll cycle, transcribes the remaining tail, then calls `_set_state("done")`.

No new public API on `AudioRecorder`. The loop reads `_recorder._buffer` under `_recorder._lock`.

### Constants

```python
CHUNK_SECONDS = 30      # unchanged — used inside transcribe_segments() internally
LIVE_CHUNK_SECONDS = 10 # new — chunk size for live loop polling
```

`transcribe_segments()` is unchanged. When called with a 10-second chunk, its internal 30-second slicing loop runs exactly once — no behavioral difference.

---

## AudioRecorder.stop() — simplified

`stop()` closes the stream and returns `None`. WAV writing is removed from `stop()`.

`_save_wav()` private method is kept (unchanged) — `test_recorder.py` calls it directly.

```python
def stop(self) -> None:
    if self._stream:
        self._stream.stop()
        self._stream.close()
        self._stream = None
```

---

## State Machine

Old states `stopped` and `transcribing` are removed. New state `finalizing` added.

| State | Start | Stop | Status |
|---|---|---|---|
| `idle` | green / enabled | disabled | "Ready" |
| `recording` | disabled | red / enabled | "Recording…" |
| `finalizing` | disabled | disabled | "Finishing…" |
| `done` | green / enabled | disabled | "Done" |

**Transitions:**
- `idle → recording`: Start clicked → mic opens, live loop thread starts
- `recording → finalizing`: Stop clicked → stream closes, stop_event set
- `finalizing → done`: live loop processes tail, fires `_set_state("done")`
- Any → `idle` on error

---

## `_live_loop(stop_event)`

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
    # Transcribe tail
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

---

## `_on_start` and `_on_stop`

```python
def _on_start(self):
    self._set_state("recording")
    self._live_stop = threading.Event()
    try:
        self._recorder.start()
        threading.Thread(target=self._live_loop, args=(self._live_stop,), daemon=True).start()
    except Exception as e:
        self._set_error(str(e))

def _on_stop(self):
    self._set_state("finalizing")
    self._recorder.stop()
    self._live_stop.set()
```

---

## Removed

| Item | Reason |
|---|---|
| `self._btn_transcribe` widget | Transcribe button removed |
| `_on_transcribe()` | No longer needed |
| `_transcribe_async()` | Replaced by `_live_loop()` |
| `self._wav_path` | No WAV file in new flow |
| States `stopped`, `transcribing` | Replaced by `recording`, `finalizing` |

---

## GUI Layout

```
[ Start ]  [ Stop ]
          Recording…
┌─────────────────────────────────────┐
│ live transcript text appears here…  │
└─────────────────────────────────────┘
[ Clear ]  [ Save File ]
```

Clear and Save File per `2026-06-11-whisper-clear-save-design.md`.

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| Mic permission denied | `_set_error(str(e))` → idle |
| Tail audio < 0.5 s | Silently skipped — not an error |
| Exception inside live loop | `self.after(0, self._set_error, str(e))` → idle |

---

## Tests

No new automated tests. Existing 6 tests call `_save_wav()`, `transcribe_segments()`, and `_ensure_offline()` directly — all unaffected by App class changes or the `stop()` return-type change.

Clear and Save File: GUI-only, verified manually.

---

## Files Changed

| File | Change |
|---|---|
| `whisper_app/app.py` | Add `LIVE_CHUNK_SECONDS`; simplify `AudioRecorder.stop()`; rewrite `App` (remove Transcribe button, add `_live_loop`, 4-state machine, Clear/Save buttons) |
| `whisper_app/tests/` | No changes |
