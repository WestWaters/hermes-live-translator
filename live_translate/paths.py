"""Where the plugin and its engine live on disk."""
from __future__ import annotations

import os
from pathlib import Path


PLUGIN = "live-translate"

HERE = Path(__file__).resolve().parent.parent      # the plugin's root folder

# Models the engine downloads from Hugging Face on first use (all Apache-2.0 / MIT).
MODELS = [
    ("mlx-community/Qwen3-ASR-0.6B-8bit", "speech recognition, 30 languages + 22 Chinese dialects (Apache-2.0)"),
    ("mlx-community/whisper-large-v3-turbo", "language identification + recognition, 99 languages (MIT)"),
    ("mlx-community/Qwen3.5-4B-MLX-4bit", "translation, 201 languages (Apache-2.0)"),
    ("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit", "own-voice speech, 10 languages (Apache-2.0)"),
    ("mlx-community/chatterbox-4bit", "own-voice speech, 23 languages (MIT)"),
    ("mlx-community/S3TokenizerV2", "voice tokenizer for chatterbox (MIT)"),
    ("mlx-community/OmniVoice-bf16", "natural voice, 600+ languages (CC-BY-NC-4.0, 'best' preset only)"),
]

def data_dir() -> Path:
    try:
        from plugins.plugin_storage import plugin_data_dir
        d = Path(plugin_data_dir(PLUGIN))
    except Exception:
        d = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")) / "plugin-data" / PLUGIN
    d.mkdir(parents=True, exist_ok=True)
    return d

def venv_python() -> Path:
    override = os.environ.get("LIVE_TRANSLATE_PYTHON")      # developers: an interpreter that already has the engine
    if override:
        return Path(override)
    v = data_dir() / "venv"
    return v / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

def is_set_up() -> bool:
    if os.environ.get("LIVE_TRANSLATE_PYTHON"):
        return venv_python().exists()
    return venv_python().exists() and (data_dir() / "venv" / ".ready").exists()
