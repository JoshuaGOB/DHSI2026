import os
import threading
import tempfile
import tkinter as tk
from tkinter import scrolledtext

import numpy as np
import sounddevice as sd
import soundfile as sf
import mlx_whisper


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

    def stop(self) -> str:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        return self._save_wav()

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

    def transcribe(self, wav_path: str) -> str:
        try:
            result = mlx_whisper.transcribe(wav_path, path_or_hf_repo=self.MODEL)
            return result.get("text", "").strip()
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)


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
