"""Voice workers: a voice model running in its own process and Python environment.

Voice models pin conflicting library versions (transformers 4.57 vs 5.x, old torch pins that do not run on new
GPUs), so each one lives in its own environment and talks to the engine over a tiny line protocol:

  engine -> worker   {"op": "ref", "wav": "<16 kHz wav path>", "text": "<its transcript>"}
                     {"op": "say", "text": "...", "lang": "es", "out": "<wav path to write>"}
  worker -> engine   {"ok": true, ...} or {"ok": false, "error": "..."}   (one JSON line per request)

A worker is started as `<python> -m lt_engine.workers.<name> [options]`. Users can point at their own program the
same way: `worker:<command line>` in a voice route, and implement the three-line protocol.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading

import numpy as np


class WorkerVoice:
    """Engine side of a voice worker (a lt_engine.voices.Voice)."""

    def __init__(self, command, name=None):
        self.command = command if isinstance(command, list) else shlex.split(command, posix=os.name != "nt")
        self.name = name or os.path.basename(self.command[-1])
        self.proc, self.lock = None, threading.Lock()
        self.tmp = tempfile.mkdtemp(prefix="lt-voice-")
        self.ref = None

    def _start(self):
        if self.proc is None or self.proc.poll() is not None:
            here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            from .childenv import minimal_env                     # voice workers need no keys at all
            env = minimal_env({"PYTHONPATH": here + os.pathsep + os.environ.get("PYTHONPATH", ""), "PYTHONUNBUFFERED": "1",
                               "PYTHONUTF8": "1"})
            self.proc = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=sys.stderr, text=True, bufsize=1, encoding="utf-8", errors="replace", env=env)
            ready = json.loads(self.proc.stdout.readline() or "{}")
            if not ready.get("ok"):
                raise RuntimeError(f"voice worker {self.name} failed to start: {ready.get('error')}")
            if self.ref is not None:
                self._call({"op": "ref", **self.ref})

    def _call(self, msg):
        with self.lock:
            self.proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
        r = json.loads(line or '{"ok": false, "error": "worker exited"}')
        if not r.get("ok"):
            raise RuntimeError(f"{self.name}: {r.get('error')}")
        return r

    def warm(self, lang):
        self._start()
        placeholder = self.ref is None                 # cloning workers need a voice before they can speak: warm up
        if placeholder:                                # on 5 s of faint noise (the length real samples have)
            self.set_reference((np.random.default_rng(0).standard_normal(16000 * 5) * 0.01).astype(np.float32), "")
        try:
            for _ in self.speak("Hello, this is a short warm-up sentence.", lang):
                pass
        finally:
            if placeholder:
                self.ref = None

    def set_reference(self, audio16k, text):
        import soundfile as sf
        path = os.path.join(self.tmp, "ref.wav")
        sf.write(path, np.asarray(audio16k, np.float32), 16000)
        self.ref = {"wav": path, "text": text or ""}
        if self.proc is not None and self.proc.poll() is None:
            self._call({"op": "ref", **self.ref})

    def speak(self, text, lang):
        from .voices import _read_wav
        self._start()
        out = os.path.join(self.tmp, "out.wav")
        self._call({"op": "say", "text": text, "lang": lang, "out": out})
        yield _read_wav(out)


def serve(load, ref, say):
    """Worker side. `load()` builds the model; `ref(model, wav16k, sr, text)` stores the speaker's voice;
    `say(model, text, lang) -> (float32 audio, sample rate)`. Runs the line protocol on stdin/stdout."""
    import soundfile as sf
    out, sys.stdout = sys.stdout, sys.stderr          # libraries' prints must not corrupt the protocol
    try:
        model = load()
        out.write(json.dumps({"ok": True}) + "\n")
        out.flush()
    except Exception as e:                           # noqa: BLE001
        out.write(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"[:500]}) + "\n")
        out.flush()
        return
    for line in sys.stdin:
        try:
            m = json.loads(line)
            if m["op"] == "ref":
                a, sr = sf.read(m["wav"], dtype="float32")
                ref(model, a, sr, m.get("text", ""))
                r = {"ok": True}
            elif m["op"] == "say":
                a, sr = say(model, m["text"], m["lang"])
                sf.write(m["out"], np.asarray(a, np.float32).reshape(-1), sr)
                r = {"ok": True, "seconds": round(len(np.asarray(a).reshape(-1)) / sr, 2)}
            else:
                r = {"ok": False, "error": f"unknown op {m.get('op')}"}
        except Exception as e:                       # noqa: BLE001
            r = {"ok": False, "error": f"{type(e).__name__}: {e}"[:500]}
        out.write(json.dumps(r) + "\n")
        out.flush()
