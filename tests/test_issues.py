"""Regression tests for the GitHub issues (WestWaters/hermes-live-translator)."""
import io
import json
import sys
import types

import pytest

from live_translate import handlers
from lt_engine import childenv, worker


# --- #1: Windows voice workers need USERNAME, and a worker's error names the real cause ---------------------------

def test_windows_workers_keep_username():
    base = {"PATH": r"C:\Windows", "USERNAME": "pascal", "USERPROFILE": r"C:\Users\pascal", "OPENAI_API_KEY": "sk-x"}
    env = childenv.minimal_env(base=base)
    assert env["USERNAME"] == "pascal"            # getpass.getuser() finds it before falling through to `import pwd`
    assert "OPENAI_API_KEY" not in env


def _wrapped():
    try:
        raise ModuleNotFoundError("No module named 'pwd'")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Could not import module 'AutoModel'. Are this object's requirements defined "
                                  "correctly?") from e


def test_worker_error_names_the_root_cause():
    with pytest.raises(ModuleNotFoundError) as e:
        _wrapped()
    why = worker._why(e.value)
    assert why.startswith("ModuleNotFoundError: No module named 'pwd'")
    assert "AutoModel" in why
    assert worker._why(ValueError("plain")) == "ValueError: plain"


def test_a_worker_that_fails_to_load_reports_the_cause(monkeypatch, capsys):
    # the plugin's CI has no engine audio stack: serve() imports soundfile first, and a failed load never touches it
    monkeypatch.setitem(sys.modules, "soundfile", types.ModuleType("soundfile"))
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    worker.serve(_wrapped, None, None)
    line = json.loads(capsys.readouterr().out.strip())
    assert line["ok"] is False and "No module named 'pwd'" in line["error"]


# --- #2: the app's start honours the plugin's `voices` setting ----------------------------------------------------

ROUTE = "en,es,fr,hi,it,pt,ja,zh=kokoro; *=system"


@pytest.fixture
def app_start(monkeypatch):
    started = []
    monkeypatch.setattr(handlers, "detect_models", lambda hermes_llm=False: {
        "translator": [{"id": ""}], "voices": [{"id": "commercial"}], "recognizer": [{"id": "auto"}]})
    monkeypatch.setattr(handlers, "can_use_hermes_llm", lambda ctx: False)
    monkeypatch.setattr(handlers.SESSION, "start", lambda *a, **k: started.append(k) or {"ok": True})
    ctx = types.SimpleNamespace(get_config=lambda key, default=None: {"voices": ROUTE}.get(key, default))
    return handlers._actions(ctx)["start"], started


def test_app_start_uses_the_plugin_voices_setting(app_start):
    start, started = app_start
    assert start({"target_language": "fr"})["ok"]
    assert started[-1]["voices"] == ROUTE


def test_the_windows_pick_still_wins(app_start):
    start, started = app_start
    assert start({"target_language": "fr", "voices": "commercial"})["ok"]
    assert started[-1]["voices"] == "commercial"


def test_a_custom_route_in_the_request_is_still_refused(app_start):
    start, started = app_start
    r = start({"target_language": "fr", "voices": "worker:/tmp/evil"})
    assert not r["ok"] and not started
