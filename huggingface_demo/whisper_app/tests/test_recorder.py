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
