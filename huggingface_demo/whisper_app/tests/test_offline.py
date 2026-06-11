import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app import _ensure_offline


def test_sets_env_when_model_dir_exists(tmp_path):
    hub_dir = tmp_path / "hub"
    hub_dir.mkdir()
    (hub_dir / "models--mlx-community--whisper-large-v3-turbo").mkdir()

    prev = os.environ.pop("HF_HUB_OFFLINE", None)
    try:
        _ensure_offline(cache_dir=hub_dir)
        assert os.environ.get("HF_HUB_OFFLINE") == "1"
    finally:
        os.environ.pop("HF_HUB_OFFLINE", None)
        if prev is not None:
            os.environ["HF_HUB_OFFLINE"] = prev


def test_does_not_set_env_when_cache_missing(tmp_path):
    prev = os.environ.pop("HF_HUB_OFFLINE", None)
    try:
        _ensure_offline(cache_dir=tmp_path / "nonexistent")
        assert "HF_HUB_OFFLINE" not in os.environ
    finally:
        os.environ.pop("HF_HUB_OFFLINE", None)
        if prev is not None:
            os.environ["HF_HUB_OFFLINE"] = prev
