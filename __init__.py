"""Live Translate for Hermes Agent: real-time speech-to-speech translation, on this computer.

This file is only the Hermes entry point. The plugin lives in live_translate/ (session, setup, model detection,
local server, state machine), the engine in lt_engine/ (its own Python environment), the window in app/ and the
desktop tray app in tray/."""
from __future__ import annotations

import atexit
from pathlib import Path

try:                                            # loaded by Hermes as the plugin package
    from .live_translate import schemas
    from .live_translate.cli import _cli, _cli_setup, _slash
    from .live_translate.handlers import make_handlers
    from .live_translate.session import SESSION
except ImportError:                             # loaded as a plain module (test runners)
    from live_translate import schemas
    from live_translate.cli import _cli, _cli_setup, _slash
    from live_translate.handlers import make_handlers
    from live_translate.session import SESSION

TOOLSET = "live-translate"


def register(ctx):
    handlers = make_handlers(ctx)
    for name, schema in (("translate_start", schemas.TRANSLATE_START), ("translate_stop", schemas.TRANSLATE_STOP),
                         ("translate_status", schemas.TRANSLATE_STATUS),
                         ("translate_languages", schemas.TRANSLATE_LANGUAGES)):
        ctx.register_tool(name=name, toolset=TOOLSET, schema=schema, handler=handlers[name])
    ctx.register_command("translate", lambda raw: _slash(ctx, raw),
                         description="Live speech translation: /translate <language> | status | stop | languages",
                         args_hint="<language> [from <language>] | status | stop")
    skills = Path(__file__).parent / "skills"            # how the agent should use the tools
    for child in sorted(skills.iterdir()) if skills.is_dir() else []:
        if (child / "SKILL.md").exists() and hasattr(ctx, "register_skill"):
            ctx.register_skill(child.name, child / "SKILL.md")
    if hasattr(ctx, "register_hook"):                  # the engine never outlives the Hermes that started it
        ctx.register_hook("on_session_finalize", lambda **_: SESSION.stop() if SESSION.running else None)
    atexit.register(lambda: SESSION.stop() if SESSION.running else None)
    ctx.register_cli_command(name="translate", help="Live speech-to-speech translation", setup_fn=_cli_setup,
                             handler_fn=_cli)
