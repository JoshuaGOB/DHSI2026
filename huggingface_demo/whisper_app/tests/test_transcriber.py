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
