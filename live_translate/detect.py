"""Find what each user can translate with: built-in models, local servers, subscriptions, Hermes's own model."""
from __future__ import annotations

import json
import sys

from .paths import HERE


# Cloud services users may already pay for. Each key is declared in plugin.yaml (optional_env) and read only through
# Hermes's secret scope (catalog rule 11); the engine process gets just the one its chosen service needs.  (env var, label, translator spec, voice spec, recognizer spec)
SUBSCRIPTIONS = [
    ("OPENAI_API_KEY", "OpenAI", "openai:https://api.openai.com/v1#gpt-5-mini", "openai:https://api.openai.com/v1#gpt-4o-mini-tts#alloy",
     "*=openai:https://api.openai.com/v1#gpt-4o-transcribe"),
    ("GEMINI_API_KEY", "Google Gemini", "openai:https://generativelanguage.googleapis.com/v1beta/openai#gemini-2.5-flash", None, None),
    ("OPENROUTER_API_KEY", "OpenRouter", "openai:https://openrouter.ai/api/v1#google/gemini-2.5-flash", None, None),
    ("GROQ_API_KEY", "Groq", "openai:https://api.groq.com/openai/v1#llama-3.3-70b-versatile", None,
     "*=openai:https://api.groq.com/openai/v1#whisper-large-v3-turbo"),
    ("ANTHROPIC_API_KEY", "Anthropic", "openai:https://api.anthropic.com/v1#claude-haiku-4-5", None, None),
    ("ELEVENLABS_API_KEY", "ElevenLabs", None, "*=elevenlabs:eleven_flash_v2_5", None),
]

LOCAL_SERVERS = [("Ollama", "http://127.0.0.1:11434/v1"), ("LM Studio", "http://127.0.0.1:1234/v1"),
                 ("llama-server", "http://127.0.0.1:8080/v1"), ("vLLM", "http://127.0.0.1:8000/v1")]

def _secret(name):
    """A key the user gave Hermes (profile-scoped). Never read from files or the raw environment."""
    try:
        from agent.secret_scope import get_secret
        return get_secret(name) or ""
    except Exception:
        return ""


def _hermes_config():
    """Hermes's config as the active profile sees it (managed overlay included), read-only."""
    try:
        from hermes_cli.config import load_config_readonly
        return load_config_readonly() or {}
    except Exception:
        return {}


def _get_json(url, timeout=0.6):
    import urllib.request
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)

def detect_models(hermes_llm=False):
    """Everything this user can translate with, found automatically: the built-in models for this hardware,
    local model servers that are running, subscriptions whose keys Hermes holds, and (when the plugin can call it)
    the model Hermes itself uses. Never sends a key anywhere."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from lt_engine.hw import detect
    hw = detect()
    out = {"hardware": hw, "recognizer": [], "translator": [], "voices": []}
    builtin = {"mlx": "Apple Silicon", "cuda": "NVIDIA GPU", "cpu": "this computer"}[hw]
    out["recognizer"].append({"id": "auto", "label": f"Built-in · Qwen3-ASR + Whisper ({builtin})", "group": "On this computer"})
    out["translator"].append({"id": "", "label": f"Built-in · Qwen3.5-4B, 201 languages ({builtin})", "group": "On this computer"})
    out["voices"] += [{"id": "commercial", "label": "Built-in · their own voice (23 languages) · free for any use", "group": "On this computer"},
                      {"id": "best", "label": "Built-in · + natural voices in 600+ languages (non-commercial)", "group": "On this computer"}]
    for name, base in LOCAL_SERVERS:                         # running local model servers
        try:
            data = _get_json(base + "/models")
        except Exception:
            continue
        for m in (data.get("data") or data.get("models") or [])[:12]:
            mid = m.get("id") or m.get("name") or m.get("model")
            if mid:
                out["translator"].append({"id": f"openai:{base}#{mid}", "label": f"{name} · {mid}", "group": "Running on this computer"})
    if hermes_llm:                                           # the model Hermes itself uses, through ctx.llm
        name = ((_hermes_config().get("model") or {}).get("default")) or "your current model"
        out["translator"].append({"id": "hermes", "label": f"My Hermes model · {name}", "group": "Your subscriptions"})
    for env, label, mt, voice, asr in SUBSCRIPTIONS:        # subscriptions: only those with a key present
        if not _secret(env):
            continue
        if mt:
            out["translator"].append({"id": mt, "label": f"{label} · {mt.split('#')[-1]}", "group": "Your subscriptions", "key": env})
        if voice:
            out["voices"].append({"id": voice, "label": f"{label} voices", "group": "Your subscriptions", "key": env})
        if asr:
            out["recognizer"].append({"id": asr, "label": f"{label} · {asr.split('#')[-1]}", "group": "Your subscriptions", "key": env})
    out["missing"] = [label for env, label, *_ in SUBSCRIPTIONS if not _secret(env)]
    return out

def hermes_skin():
    """The skin the user picked in Hermes (display.skin), so the window can match it."""
    return {"skin": ((_hermes_config().get("display") or {}).get("skin")) or "default"}


def languages():
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from lt_engine import langs                       # plain data, no engine dependencies
    rows = []
    for code, name in sorted(langs.NAMES.items(), key=lambda kv: kv[1]):
        voice = "own voice" if (code in langs.FAST_CLONE or code in langs.CLONE_LANGS) else "system voice"
        rows.append({"code": code, "name": name, "fast_recognition": code in langs.FAST_ASR, "voice": voice})
    return {"understands": len(rows), "translates_between": "201 languages and dialects",
            "speaks_in_own_voice": sum(r["voice"] == "own voice" for r in rows), "languages": rows}

def keys_for(translator, voices, recognizer):
    """Hand the engine exactly the keys its chosen services need, under the names it reads."""
    env = {}
    for spec, target in ((translator, "LIVE_TRANSLATE_API_KEY"), (voices, "LIVE_TRANSLATE_TTS_KEY"),
                         (recognizer, "LIVE_TRANSLATE_STT_KEY")):
        spec = spec or ""
        for envname, _label, *_ in SUBSCRIPTIONS:
            host = {"OPENAI_API_KEY": "api.openai.com", "GEMINI_API_KEY": "generativelanguage.googleapis.com",
                    "OPENROUTER_API_KEY": "openrouter.ai", "GROQ_API_KEY": "api.groq.com",
                    "ANTHROPIC_API_KEY": "api.anthropic.com"}.get(envname)
            if host and host in spec and _secret(envname):
                env[target] = _secret(envname)
        if "elevenlabs:" in spec and _secret("ELEVENLABS_API_KEY"):
            env["ELEVENLABS_API_KEY"] = _secret("ELEVENLABS_API_KEY")
    return env
