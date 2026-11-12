"""/translate in chat and `hermes translate ...` on the command line."""
from __future__ import annotations

import json
import os
import time

from .detect import languages
from .handlers import make_handlers, open_window
from .install import setup
from .paths import HERE, is_set_up, venv_python, child_env
from .session import SESSION


def _slash(ctx, raw: str):
    """/translate <language> [from <language>] | /translate stop | /translate status"""
    words = (raw or "").split()
    h = make_handlers(ctx)
    if not words or words[0] == "status":
        st = json.loads(h["translate_status"]({"captions": 5}))
        if not st.get("running"):
            return "Live translation is off. Start it with /translate <language>, e.g. /translate Spanish"
        lines = [f"Translating into {st['target']} ({st.get('detected') or 'detecting'}), state: {st['state']}, "
                 f"median delay {st.get('delay_median_s')} s"]
        lines += [f"  {c['text']}  →  {c['translation']}" for c in st.get("captions", [])]
        return "\n".join(lines)
    if words[0] == "stop":
        r = json.loads(h["translate_stop"]({}))
        return f"Live translation stopped ({r.get('phrases', 0)} phrases, median delay {r.get('delay_median_s')} s)."
    if words[0] == "languages":
        r = json.loads(h["translate_languages"]({}))
        return (f"Understands {r['understands']} languages, translates between {r['translates_between']}, "
                f"speaks {r['speaks_in_own_voice']} in the speaker's own voice.")
    args = {"target_language": words[0]}
    if len(words) >= 3 and words[1] == "from":
        args["source_language"] = words[2]
    r = json.loads(h["translate_start"](args))
    if not r.get("ok"):
        return f"Could not start: {r.get('error')}"
    return (f"Live translation into {r['target']} is starting ({r['state']}). Input: {r['input']}; output: "
            f"{r['output']}. /translate status for captions, /translate stop to end.")


def _cli_setup(sub):
    subs = sub.add_subparsers(dest="translate_cmd")
    subs.add_parser("setup", help="create the engine environment and download its models (~6 GB, once)")
    p = subs.add_parser("start", help="translate live audio (runs until Ctrl-C)")
    p.add_argument("--to", required=True, help="language to hear (code or name)")
    p.add_argument("--from", dest="src", help="language being spoken (default: detect)")
    p.add_argument("--input", help="input device (default microphone; 'BlackHole 2ch' = meeting audio)")
    p.add_argument("--output", help="output device (default speakers; 'BlackHole 2ch' = virtual mic)")
    p.add_argument("--no-clone", action="store_true", help="do not use the speaker's own voice")
    f = subs.add_parser("file", help="translate an audio/video file as if it were live")
    f.add_argument("path")
    f.add_argument("--to", required=True)
    f.add_argument("--from", dest="src")
    f.add_argument("--output", help="output device (default speakers)")
    f.add_argument("--record", help="also save original + translation to this .wav")
    f.add_argument("--no-clone", action="store_true")
    subs.add_parser("serve", help="run the local control server for the tray app / Hermes Desktop (Ctrl-C to stop)")
    subs.add_parser("languages", help="list supported languages")
    subs.add_parser("devices", help="list audio devices")


def _cli(args):
    import subprocess
    cmd = getattr(args, "translate_cmd", None)
    if cmd == "setup":
        setup()
    elif cmd in ("start", "file"):
        if cmd == "start":
            r = SESSION.start(args.to, args.src, args.input, args.output, not args.no_clone)
        else:
            r = SESSION.start(args.to, args.src, None, args.output, not args.no_clone,
                                    input_file=os.path.abspath(args.path),
                                    record=args.record and os.path.abspath(args.record))
        if not r.get("ok"):
            print(r.get("error"))
            return
        print(f"starting: {r['state']}  (Ctrl-C to stop)")
        if r.get("captions_page"):
            print(f"live captions page: {r['captions_page']}  (OBS: add as Browser Source)")
        shown = 0
        try:
            while SESSION.running:
                st = SESSION.status(200)
                for c in st["captions"][shown:]:
                    print(f"[{c['src']}] {c['text']}\n[{c['tgt']}] {c['translation']}\n")
                shown = len(st["captions"])
                time.sleep(0.3)
        except KeyboardInterrupt:
            pass
        r = SESSION.stop()
        print(f"stopped: {r.get('phrases')} phrases, median delay {r.get('delay_median_s')} s; "
              f"transcript {r.get('transcript')}")
    elif cmd == "serve":
        print(f"Live Translate control window: {open_window()}  (Ctrl-C to stop)", flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            SESSION.stop()
    elif cmd == "languages":
        r = languages()
        for row in r["languages"]:
            print(f"{row['code']:>4}  {row['name']:<18} {row['voice']}")
        print(f"\n{r['understands']} languages understood; {r['speaks_in_own_voice']} spoken in the speaker's voice")
    elif cmd == "devices":
        if not is_set_up():
            print("run `hermes translate setup` first")
            return
        out = subprocess.run([str(venv_python()), "-m", "lt_engine", "--list-devices"], cwd=str(HERE),
                             env=child_env({"PYTHONPATH": str(HERE)}), capture_output=True, text=True).stdout
        for d in json.loads(out)["devices"]:
            kind = "in " if d["inputs"] else "   "
            kind += "out" if d["outputs"] else ""
            print(f"{d['index']:>3}  {kind:<6} {d['name']}")
    else:
        print("usage: hermes translate {setup,start,languages,devices}")
