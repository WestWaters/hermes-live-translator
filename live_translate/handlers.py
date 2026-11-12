"""What the agent's tools and the app window call."""
from __future__ import annotations

import json
import threading

from .detect import detect_models, hermes_skin, keys_for, languages, _secret
from . import obs_settings
from .install import setup
from .session import SESSION, devices


# ------------------------------------------------------------------ tool handlers (return JSON strings)
def _cfg(ctx, key, default):
    try:
        return ctx.get_config(key, default=default) if ctx is not None else default
    except Exception:
        return default

_CTX = None                     # the plugin context Hermes gave register(); the window's actions use it too


def bind(ctx):
    global _CTX
    _CTX = ctx


def _secrets():
    """The keys declared in plugin.yaml, read through Hermes (profile-scoped); handed to the engine process only."""
    out = {k: _secret(k) for k in ("ELEVENLABS_API_KEY", "LIVE_TRANSLATE_API_KEY")}
    if out["LIVE_TRANSLATE_API_KEY"]:                       # one key for a self-chosen OpenAI-compatible service
        out["LIVE_TRANSLATE_TTS_KEY"] = out["LIVE_TRANSLATE_STT_KEY"] = out["LIVE_TRANSLATE_API_KEY"]
    return {k: v for k, v in out.items() if v}


def can_use_hermes_llm(ctx=None):
    return getattr(ctx or _CTX, "llm", None) is not None


def hermes_llm(body):
    """Translate one request with the model Hermes is using (ctx.llm: Hermes owns the provider and the key).
    Takes and returns the OpenAI chat-completions shape, so the engine talks to it like any other server."""
    ctx = _CTX
    if not can_use_hermes_llm(ctx):
        raise RuntimeError("Hermes's model is not available to plugins in this process")
    r = ctx.llm.complete(messages=body.get("messages") or [], max_tokens=int(body.get("max_tokens") or 256),
                         temperature=0.0, purpose="live-translate.translate")
    return {"object": "chat.completion", "model": "hermes",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": r.text or ""},
                         "finish_reason": "stop"}]}


LOCAL_SURFACES = ("", "cli", "tui", "desktop", "acp")     # someone is at this computer (or no session at all)


def _not_here(ctx):
    """Translation opens this computer's microphone and speakers. From a cron job or a chat app (Telegram, ...)
    nobody is at the computer, so starting is refused unless the user allowed it (catalog rule 12)."""
    if ctx is None or _cfg(ctx, "allow_remote_start", False):
        return ""
    unknown = ("not started: this Hermes cannot say where the request came from (it needs a Hermes with "
               "session_surface), and live translation opens this computer's microphone and speakers. Start it from "
               "the CLI, TUI or Desktop of a current Hermes, or turn on 'Allow starting from chat apps'.")
    if not hasattr(ctx, "session_surface"):                   # older Hermes: fail closed (catalog rule 12)
        return unknown
    try:
        if hasattr(ctx, "current_cron_execution") and ctx.current_cron_execution() is not None:
            return "not started: a scheduled job cannot open this computer's microphone"
        surface = ctx.session_surface()
    except Exception:
        return unknown
    if surface not in LOCAL_SURFACES:
        return (f"not started: this request came from {surface}, and live translation uses the microphone and "
                "speakers of the computer Hermes runs on. Start it there (CLI, TUI or Desktop), or turn on "
                "'Allow starting from chat apps' in the plugin settings.")
    return ""


def make_handlers(ctx=None):
    if ctx is not None:
        bind(ctx)

    def start(args, **_):
        why = _not_here(ctx)
        if why:
            return json.dumps({"ok": False, "error": why}, ensure_ascii=False)
        target = args.get("target_language") or _cfg(ctx, "target_language", "en")
        mt, vo, rec = (_cfg(ctx, "translator", "") or None, _cfg(ctx, "voices", "commercial") or None,
                       _cfg(ctx, "recognizer", "auto") or None)
        mode = args.get("mode") or ("call" if args.get("two_way") else "listen")
        inp = args.get("input_device") or _cfg(ctx, "input_device", "") or None
        out = args.get("output_device") or _cfg(ctx, "output_device", "") or None
        to_them = args.get("to_them")
        if mode == "call" and not (inp and to_them):        # "Hey Hermes, video call": find the virtual cables
            found = call_devices(devices().get("devices", []))
            if "error" in found:
                return json.dumps({"ok": False, "error": found["error"]}, ensure_ascii=False)
            inp, to_them = inp or found["call_in"], to_them or found["to_them"]
        r = SESSION.start(target, args.get("source_language"), inp, out,
                          args.get("voice_clone", _cfg(ctx, "voice_clone", True)),
                          voices=vo, recognizer=rec, translator=mt, secrets=keys_for(mt, vo, rec),
                          two_way=mode == "call", to_them=to_them, play=mode != "stream", obs=mode == "stream")
        if r.get("ok"):
            r["mode"] = mode
            if mode == "stream" and SESSION.obs_feed is None:
                r["note"] = ("captions are on the caption page; for OBS closed captions, turn on OBS's WebSocket "
                             "server and save its password under Settings > Stream")
        return json.dumps(r, ensure_ascii=False)

    def stop(args, **_):
        return json.dumps(SESSION.stop(), ensure_ascii=False)

    def status(args, **_):
        return json.dumps(SESSION.status(int(args.get("captions") or 10)), ensure_ascii=False)

    def langs_(args, **_):
        return json.dumps(languages(), ensure_ascii=False)

    return {"translate_start": start, "translate_stop": stop, "translate_status": status,
            "translate_languages": langs_}

VIRTUAL = ("blackhole", "cable", "vb-audio", "loopback", "monitor", "virtual", "soundflower", "voicemeeter")


def call_devices(devs):
    """The two virtual audio devices a translated call needs: one carrying the meeting's audio in (call_in) and another
    that the meeting app uses as its microphone (to_them). One cable for both would loop the call into itself."""
    virt = [d for d in devs if any(v in d.get("name", "").lower() for v in VIRTUAL)]
    ins = [d["name"] for d in virt if d.get("inputs", 0) > 0]
    outs = [d["name"] for d in virt if d.get("outputs", 0) > 0]
    for i in ins:
        for o in outs:
            if o != i:
                return {"call_in": i, "to_them": o}
    have = ", ".join(sorted({d["name"] for d in virt})) or "none"
    return {"error": ("a translated call needs two virtual audio devices (the meeting's audio in, and a microphone "
                      "for the meeting app): macOS: BlackHole 2ch + BlackHole 16ch; Windows: VB-CABLE A + B; Linux: two "
                      f"PipeWire null sinks. Found: {have}.")}


def _actions(ctx=None):
    """What the control window may do (server.py's /api routes)."""
    ctx = ctx or _CTX

    def start(b):
        # The window (a local web page) may only pick what detect_models() found or a preset. Custom routes --
        # command:/worker: voices, arbitrary service URLs -- come from the plugin's own config, never a request.
        found = detect_models(hermes_llm=can_use_hermes_llm(ctx))
        allowed = {kind: {m["id"] for m in found.get(kind, [])} for kind in ("translator", "voices", "recognizer")}
        allowed["voices"] |= {"commercial", "best"}
        allowed["recognizer"] |= {"auto"}
        allowed["translator"] |= {""}
        for kind in ("translator", "voices", "recognizer"):
            v = b.get(kind)
            if v is not None and v not in allowed[kind]:
                return {"ok": False, "error": f"{kind} {v!r} is not one of the models found on this computer; "
                                              "custom routes are set in the plugin's settings"}
        rec = b.get("recognizer") or _cfg(ctx, "recognizer", "auto") or None
        mt = b.get("translator") if b.get("translator") is not None else (_cfg(ctx, "translator", "") or None)
        vo = b.get("voices") or None
        return SESSION.start(b.get("target_language") or "en", b.get("source_language"), b.get("input_device"),
                             b.get("output_device"), b.get("voice_clone", True), voices=vo, recognizer=rec,
                             translator=mt or None, secrets=keys_for(mt, vo, rec),
                             two_way=bool(b.get("two_way")), my_mic=b.get("my_mic"), to_them=b.get("to_them"),
                             volume=b.get("volume", getattr(SESSION, "volume", None)))

    def setup_(b):
        threading.Thread(target=setup, daemon=True).start()
        return {"ok": True, "state": SESSION.state.snapshot()}

    return {"start": start, "stop": lambda b: SESSION.stop(), "setup": setup_, "devices": lambda b: devices(),
            "volume": lambda b: SESSION.set_volume(b.get("value", 1.0)),
            "models": lambda b: detect_models(hermes_llm=can_use_hermes_llm(ctx)), "llm": hermes_llm,
            "obs": lambda b: obs_settings.update(b),
            "languages": lambda b: languages(), "hermes-skin": lambda b: hermes_skin()}

def open_window():
    """Start the local server (if needed) and return the control window URL."""
    SESSION._overlay()
    from .server import TOKEN
    return (SESSION.overlay_url or "").rstrip("/") + "/app#t=" + TOKEN     # the token travels in the fragment only
