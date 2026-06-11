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
