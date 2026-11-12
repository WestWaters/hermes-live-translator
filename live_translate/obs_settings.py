"""The streamer's OBS caption settings, kept in the plugin's data folder (owner-only file)."""
from __future__ import annotations

import json
import os

from .paths import data_dir


def _file():
    return data_dir() / "obs.json"


def load():
    try:
        return json.loads(_file().read_text(encoding="utf-8"))
    except Exception:
        return {"enabled": False, "host": "127.0.0.1", "port": 4455, "password": ""}


def save(cfg):
    f = _file()
    f.write_text(json.dumps(cfg), encoding="utf-8")
    try:
        os.chmod(f, 0o600)
    except OSError:
        pass


def feed(force=False):
    """A caption feed when closed captions are switched on (or `force`: stream mode), else None."""
    c = load()
    if not (c.get("enabled") or force):
        return None
    from .obs import CaptionFeed
    return CaptionFeed(c.get("host", "127.0.0.1"), c.get("port", 4455), c.get("password", ""))


def update(body):
    """Change the settings; with "test", connect to OBS and report its version or the problem."""
    c = load()
    for k in ("enabled", "host", "port"):
        if k in body:
            c[k] = body[k]
    if body.get("password"):
        c["password"] = body["password"]
    save(c)
    out = {"enabled": bool(c.get("enabled")), "port": c.get("port", 4455), "has_password": bool(c.get("password"))}
    if body.get("test"):
        from .obs import OBS
        try:
            v = OBS(c.get("host", "127.0.0.1"), c.get("port", 4455), c.get("password", "")).request("GetVersion")
            out["test"] = f"Connected to OBS {v.get('obsVersion', '')}"
        except Exception as e:                                      # noqa: BLE001
            out["test"] = f"Could not reach OBS. {e} In OBS: Tools > WebSocket Server Settings > Enable."
    return out
