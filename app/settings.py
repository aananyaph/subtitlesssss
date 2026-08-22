"""Persistent application settings (JSON) for the Live Subtitle PC App.

Settings are stored at %APPDATA%\\LiveSubtitlePC\\settings.json (or
$XDG_CONFIG_HOME/LiveSubtitlePC/settings.json on POSIX), with a fallback
to a local settings.json next to the application when the config dir is
unavailable. Missing or corrupt files fall back to defaults.
"""

import json
import os
import traceback
from typing import Dict

APP_NAME = "LiveSubtitlePC"

DEFAULT_SETTINGS: Dict[str, object] = {
    "model": "distil-small.en",
    "font_size": "medium",
    "position": "bottom",
    "min_chunk_floor_sec": 1.0,
    "stale_audio_timeout_sec": 2.0,
    "device_index": None,
}


def get_settings_path() -> str:
    """Returns the absolute path of the settings JSON file."""
    override = os.environ.get("LIVE_SUBTITLE_SETTINGS")
    if override:
        return override

    base = os.environ.get("APPDATA")
    if not base:
        base = os.environ.get("XDG_CONFIG_HOME")
    if not base:
        base = os.path.join(os.path.expanduser("~"), ".config")

    return os.path.join(base, APP_NAME, "settings.json")


def load_settings() -> Dict[str, object]:
    """Loads settings from disk, merging with defaults."""
    settings = dict(DEFAULT_SETTINGS)
    path = get_settings_path()
    if not os.path.exists(path):
        return settings

    try:
        with open(path, "r", encoding="utf-8") as f:
            stored = json.load(f)
        if isinstance(stored, dict):
            for k, v in stored.items():
                if k in DEFAULT_SETTINGS:
                    settings[k] = v
    except Exception:
        traceback.print_exc()

    return settings


def save_settings(settings: Dict[str, object]) -> None:
    """Persists settings to disk as JSON."""
    path = get_settings_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
    except Exception:
        traceback.print_exc()
