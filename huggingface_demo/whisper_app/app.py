import os
from pathlib import Path
import threading
import tempfile
import tkinter as tk
from tkinter import scrolledtext, filedialog

import numpy as np
import sounddevice as sd
import soundfile as sf
import mlx_whisper


def _ensure_offline(cache_dir: Path = None) -> None:
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if cache_dir.exists() and any(
        d.name.startswith("models--mlx-community--whisper-large-v3-turbo")
        for d in cache_dir.iterdir()
        if d.is_dir()
    ):
        os.environ["HF_HUB_OFFLINE"] = "1"


CHUNK_SECONDS = 30
LIVE_CHUNK_SECONDS = 10


class AudioRecorder:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._buffer: list[np.ndarray] = []
        self._stream = None
        self._lock = threading.Lock()

    def start(self):
        self._buffer = []
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _callback(self, indata, frames, time, status):
        with self._lock:
            self._buffer.append(indata.copy().flatten())

    def _save_wav(self) -> str:
        with self._lock:
            audio = np.concatenate(self._buffer) if self._buffer else np.zeros((1,), dtype=np.float32)
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(path, audio, self.sample_rate)
        return path


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


class App(tk.Tk):
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

        self._set_state("idle")

    # ── button handlers ───────────────────────────────────────────────────────

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
        if self._live_stop:
            self._live_stop.set()

    def _live_loop(self, stop_event: threading.Event):
        try:
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
        except Exception as e:
            self.after(0, self._set_error, str(e))

    # ── GUI helpers ───────────────────────────────────────────────────────────

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

    def _on_clear(self):
        self._text.config(state=tk.NORMAL)
        self._text.delete("1.0", tk.END)
        self._text.config(state=tk.DISABLED)

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
