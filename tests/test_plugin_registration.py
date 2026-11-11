"""What the plugin registers with Hermes, through a context that records every call (as Hermes's own does)."""
import json
import sys

import pytest

plugin = sys.modules["live_translate_plugin"]


class RecordingCtx:
    def __init__(self, surface="cli", cron=None, settings=None):
        self.tools, self.commands, self.cli, self.skills, self.hooks = {}, {}, {}, {}, {}
        self.surface, self.cron, self.settings = surface, cron, settings or {}
        self.llm = None

    def register_tool(self, name, toolset, schema, handler, **kw):
        self.tools[name] = {"toolset": toolset, "schema": schema, "handler": handler, **kw}

    def register_command(self, name, handler, description="", args_hint=""):
        self.commands[name] = {"handler": handler, "description": description, "args_hint": args_hint}

    def register_cli_command(self, name, help, setup_fn, handler_fn):
        self.cli[name] = {"help": help, "setup_fn": setup_fn, "handler_fn": handler_fn}

    def register_skill(self, name, path, description=""):
        self.skills[name] = path

    def register_hook(self, event, fn):
        self.hooks[event] = fn

    def get_config(self, key, default=None):
        return self.settings.get(key, default)

    def session_surface(self):
        return self.surface

    def current_cron_execution(self):
        return self.cron


@pytest.fixture
def ctx():
    c = RecordingCtx()
    plugin.register(c)
    return c


def test_registers_the_four_tools_in_one_toolset(ctx):
    assert set(ctx.tools) == {"translate_start", "translate_stop", "translate_status", "translate_languages"}
    assert {t["toolset"] for t in ctx.tools.values()} == {"live-translate"}
    for name, t in ctx.tools.items():
        assert t["schema"]["name"] == name and t["schema"]["description"]


def test_manifest_lists_exactly_the_registered_tools(ctx):
    import yaml
    manifest = yaml.safe_load((plugin.Path(plugin.__file__).parent / "plugin.yaml").read_text())
    assert set(manifest["provides_tools"]) == set(ctx.tools)


def test_registers_the_slash_command_cli_skill_and_cleanup(ctx):
    assert "translate" in ctx.commands and ctx.commands["translate"]["args_hint"]
    assert "translate" in ctx.cli
    assert "live-translate" in ctx.skills and ctx.skills["live-translate"].name == "SKILL.md"
    assert "on_session_finalize" in ctx.hooks


def test_cli_subcommands(ctx):
    import argparse
    p = argparse.ArgumentParser()
    ctx.cli["translate"]["setup_fn"](p)
    sub = next(a for a in p._actions if isinstance(a, argparse._SubParsersAction))
    assert {"setup", "start", "serve", "languages", "devices"} <= set(sub.choices)


@pytest.mark.parametrize("surface,cron,allowed", [("cli", None, True), ("desktop", None, True),
                                                   ("telegram", None, False), ("cli", object(), False)])
def test_microphone_is_never_opened_from_chat_apps_or_cron(monkeypatch, surface, cron, allowed):
    from live_translate import handlers
    started = []
    monkeypatch.setattr(handlers.SESSION, "start", lambda *a, **k: started.append(1) or {"ok": True})
    c = RecordingCtx(surface=surface, cron=cron)
    r = json.loads(handlers.make_handlers(c)["translate_start"]({"target_language": "es"}))
    assert bool(started) == allowed and r["ok"] == allowed


def test_the_user_can_allow_remote_starts(monkeypatch):
    from live_translate import handlers
    monkeypatch.setattr(handlers.SESSION, "start", lambda *a, **k: {"ok": True})
    c = RecordingCtx(surface="telegram", settings={"allow_remote_start": True})
    assert json.loads(handlers.make_handlers(c)["translate_start"]({"target_language": "es"}))["ok"]


def _start(monkeypatch, args, devs=()):
    from live_translate import handlers
    seen = {}
    monkeypatch.setattr(handlers, "devices", lambda: {"devices": list(devs)})
    monkeypatch.setattr(handlers.SESSION, "start", lambda *a, **k: seen.update(args=a, kw=k) or {"ok": True})
    monkeypatch.setattr(handlers.SESSION, "obs_feed", object(), raising=False)
    r = json.loads(handlers.make_handlers(RecordingCtx())["translate_start"](args))
    return r, seen


CABLES = [{"name": "MacBook Microphone", "inputs": 1, "outputs": 0},
          {"name": "BlackHole 2ch", "inputs": 2, "outputs": 2},
          {"name": "BlackHole 16ch", "inputs": 16, "outputs": 16}]


def test_voice_command_modes(monkeypatch):
    r, s = _start(monkeypatch, {"target_language": "es"})                       # "Hey Hermes, translate this"
    assert r["mode"] == "listen" and s["kw"]["play"] and not s["kw"]["two_way"] and not s["kw"]["obs"]
    r, s = _start(monkeypatch, {"target_language": "en", "mode": "stream"})     # "Hey Hermes, stream"
    assert not s["kw"]["play"] and s["kw"]["obs"]
    r, s = _start(monkeypatch, {"target_language": "en", "source_language": "es", "mode": "call"}, CABLES)
    assert s["kw"]["two_way"] and s["args"][2] == "BlackHole 2ch" and s["kw"]["to_them"] == "BlackHole 16ch"


def test_a_call_with_one_cable_says_what_to_add(monkeypatch):
    r, s = _start(monkeypatch, {"target_language": "en", "source_language": "es", "mode": "call"}, CABLES[:2])
    assert not r["ok"] and "two virtual audio devices" in r["error"] and not s
