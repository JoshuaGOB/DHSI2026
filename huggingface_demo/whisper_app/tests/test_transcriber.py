import os
import sys
import numpy as np
import soundfile as sf
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app import WhisperTranscriber


def _make_silence_wav(duration_sec: float = 2.0, sample_rate: int = 16000) -> str:
    audio = np.zeros(int(duration_sec * sample_rate), dtype=np.float32)
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, audio, sample_rate)
    return path


def test_transcribe_returns_string():
    wav_path = _make_silence_wav()
    transcriber = WhisperTranscriber()
    result = transcriber.transcribe(wav_path)
    # wav_path already deleted by transcribe(); don't unlink again
    assert isinstance(result, str), f"Expected str, got {type(result)}"


def test_transcribe_deletes_wav():
    wav_path = _make_silence_wav()
    transcriber = WhisperTranscriber()
    transcriber.transcribe(wav_path)
    assert not os.path.exists(wav_path), "Temp WAV was not cleaned up after transcription"
