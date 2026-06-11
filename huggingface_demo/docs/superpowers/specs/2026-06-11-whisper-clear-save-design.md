# Whisper App — Clear + Save File Spec
**Date:** 2026-06-11
**Builds on:** `whisper_app/app.py` (three-button streaming app)
**Goal:** Add a Clear button to erase the transcript text area and a Save File button that opens a native macOS file-picker dialog to write the transcript to disk.

---

## Context

- Existing layout: [Start] [Stop] [Transcribe] row → status label → ScrolledText area
- Neither new button interacts with the five recording states (idle / recording / stopped / transcribing / done)
- No new dependencies — `tkinter.filedialog` is stdlib

---

## New UI Elements

A second `tk.Frame` (`action_row`) packed **below** the `ScrolledText` widget with `pady=(0, 16)`.

Two buttons inside `action_row`, packed `side=tk.LEFT`:

| Button | Width | Color | Always enabled |
|---|---|---|---|
| Clear | 10 | `#9E9E9E` grey | ✅ |
| Save File | 10 | `#2196F3` blue | ✅ |

Import added: `from tkinter import filedialog`

---

## `_on_clear()`

```python
def _on_clear(self):
    self._text.config(state=tk.NORMAL)
    self._text.delete("1.0", tk.END)
    self._text.config(state=tk.DISABLED)
```

No state machine changes. Always callable.

---

## `_on_save()`

```python
def _on_save(self):
    path = filedialog.asksaveasfilename(
        defaultextension=".txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
    )
    if not path:
        return  # user cancelled
    content = self._text.get("1.0", tk.END)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
```

- If the user cancels the dialog, nothing happens
- Writes the full current text area content (including any trailing newline from `tk.END`)
- Encoding: UTF-8 (required for Spanish/French/Portuguese transcripts)

---

## State Machine Impact

None. `_STATES` dict and `_set_state()` are unchanged. Clear and Save File are never referenced in state transitions.

---

## Files Changed

| File | Change |
|---|---|
| `whisper_app/app.py` | Add `from tkinter import filedialog`; add `action_row` frame + two buttons in `__init__`; add `_on_clear()` and `_on_save()` methods |
| `whisper_app/tests/` | No changes — Clear and Save are GUI-only; verified manually |
