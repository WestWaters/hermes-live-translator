"""One running engine: start / stop it, read its events, keep live captions and the explicit state."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from collections import deque

from .paths import HERE, child_env, data_dir, is_set_up, venv_python


class Session:
    """One running engine, its live captions, and its explicit state (state.py)."""

    def __init__(self):
        from .state import State
        self.state = State("idle" if is_set_up() else "not_set_up")
        self.proc = None
        self.lock = threading.Lock()
        self.captions = deque(maxlen=200)
        self.lags = []
        self.events = deque(maxlen=50)
        self.info = {}
        self.log_path = None
        self.overlay_url = None
        self.obs_feed = None

    def _overlay(self):
        if self.overlay_url is None:
            from . import server
            from .handlers import _actions
            self.overlay_url, self._srv = server.serve(self, actions=_actions())
        return self.overlay_url

    def _captions_page(self):
        """The caption page link for OBS / a browser: the token rides in the fragment (never sent in a request line)."""
        from .server import TOKEN
        return (self.overlay_url or "").rstrip("/") + "/#t=" + TOKEN

    @property
    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, target, source=None, input_device=None, output_device=None, voice_clone=True,
              input_file=None, record=None, voices=None, recognizer=None, translator=None, secrets=None,
              two_way=False, my_mic=None, to_them=None, volume=None, play=True, obs=False):
        if self.running:
            return {"ok": False, "error": "live translation is already running; stop it first",
                    "target": self.info.get("target"), "state": self.state.snapshot()}
        if not is_set_up():
            return {"ok": False, "error": "not set up yet: run `hermes translate setup` once (downloads ~6 GB)",
                    "state": self.state.snapshot()}
        if two_way and not source:
            return {"ok": False, "error": "two-way needs the other person's language: pick it under 'They speak'",
                    "state": self.state.snapshot()}
        if self.state.session == "not_set_up":                 # set up since this process started
            self.state.session = "idle"
        self.state.to("starting")
        cmd = [str(venv_python()), "-m", "lt_engine", "--to", target]
        cmd += ["--file", str(input_file)] if input_file else ["--mic"]
        if record:
            cmd += ["--record", str(record)]
        if source:
            cmd += ["--from", source]
        if input_device:
            cmd += ["--in-device", str(input_device)]
        if output_device:
            cmd += ["--out-device", str(output_device)]
        if not voice_clone:
            cmd.append("--no-clone")
        if voices:
            cmd += ["--voices", voices]
        if recognizer:
            cmd += ["--recognizer", recognizer]
        if translator == "hermes":                           # Hermes's own model, through this plugin (ctx.llm)
            from .server import TOKEN
            translator = "openai:" + self._overlay().rstrip("/") + "/v1#hermes"
            secrets = {**(secrets or {}), "LIVE_TRANSLATE_API_KEY": TOKEN}
        if translator:
            cmd += ["--translator", translator]
        if volume is not None:
            cmd += ["--volume", str(volume)]
        if not play:                                         # stream mode: captions only, the stream keeps its voice
            cmd.append("--no-play")
        if two_way:
            cmd.append("--two-way")
            if my_mic:
                cmd += ["--my-mic", str(my_mic)]
            if to_them:
                cmd += ["--to-them", str(to_them)]
        from .obs_settings import feed
        self.obs_feed = feed(force=obs)
        sessions = data_dir() / "sessions"
        sessions.mkdir(exist_ok=True)
        self.log_path = sessions / time.strftime("%Y%m%d-%H%M%S.jsonl")
        env = child_env({"PYTHONPATH": str(HERE), "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1",  # UTF-8 both ends
                         "LIVE_TRANSLATE_ENVS": os.environ.get("LIVE_TRANSLATE_ENVS", str(data_dir() / "envs"))},
                        keys=secrets)                                              # the selected services need
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0   # lets stop() send Ctrl-Break
        self.proc = subprocess.Popen(cmd, cwd=str(HERE), env=env, stdout=subprocess.PIPE, stdin=subprocess.PIPE,
                                     creationflags=flags,
                                     stderr=open(data_dir() / "engine.log", "a", encoding="utf-8"), text=True, bufsize=1,
                                     encoding="utf-8", errors="replace")
        self.captions.clear()
        self.lags = []
        try:
            self._overlay()
        except Exception:                                   # captions page is optional
            pass
        self.info = {"target": target, "source": source or "auto", "started": time.time(),
                     "input": input_file or input_device or "default microphone", "output": output_device or "speakers",
                     "captions_page": self._captions_page()}
        threading.Thread(target=self._read, daemon=True).start()
        return {"ok": True, **self.info, "state": self.state.snapshot()}

    def _read(self):
        st = self.state
        with open(self.log_path, "a", encoding="utf-8") as log:
            for line in self.proc.stdout:
                log.write(line)
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                kind = ev.get("event")
                with self.lock:
                    if kind == "listening" and st.session == "starting":
                        st.to("live")
                    elif kind == "activity":
                        st.facet(activity=ev.get("state", "waiting"))
                    elif kind == "heard":
                        st.facet(activity="translating", language=ev.get("lang") or "detecting")
                    elif kind == "caption":
                        self.captions.append({k: ev.get(k) for k in ("id", "lane", "speaker", "src", "tgt", "text",
                                                                     "translation")})
                        if self.obs_feed is not None and ev.get("lane", "a") == "a":
                            self.obs_feed.push(ev.get("translation") or "")
                        self.info["detected"] = ev.get("src")
                        st.facet(language=ev.get("src") or "detecting")
                    elif kind == "latency":
                        self.lags.append(ev["lag_s"])
                        st.facet(activity="speaking")
                    elif kind == "spoken":
                        st.facet(activity="waiting")
                    elif kind == "voice":
                        st.facet(voice="cloned" if ev.get("cloned") else "stock")
                    elif kind == "pace":
                        st.facet(pace=ev.get("state", "on_time"))
                    elif kind == "stopped":
                        self.stop_reason = ev.get("reason", "")
                    elif kind in ("ready", "summary", "error"):
                        self.events.append(ev)
        code = self.proc.wait()
        with self.lock:
            reason = getattr(self, "stop_reason", "")
            if code not in (0, None) or reason:
                tail = ""
                try:
                    tail = (data_dir() / "engine.log").read_text(encoding="utf-8", errors="replace")[-300:].strip().splitlines()[-1]
                except Exception:
                    pass
                st.to("error", reason or f"engine exited ({code}): {tail}")
            elif st.session in ("starting", "live", "stopping"):
                st.to("idle")
            self.stop_reason = ""

    def stop(self, timeout=30):
        if not self.running:
            # Nothing is running but the state still says it is (the engine died, or was never started):
            # Stop must always bring the app back to Ready.
            if self.state.session in ("starting", "live", "stopping"):
                if self.state.session != "stopping":
                    self.state.to("stopping")
                self.state.to("idle")
            return {"ok": True, **self.summary(), "state": self.state.snapshot()}
        if self.state.can("stopping"):
            self.state.to("stopping")
        self.proc.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
        for _ in range(50):                                  # let the reader settle the final state
            if self.state.session != "stopping":
                break
            time.sleep(0.1)
        _forget_voice_samples()
        return {"ok": True, **self.summary(), "state": self.state.snapshot()}

    def set_volume(self, value):
        """Change the translated voice's volume now (0-1.5); remembered for the next session too."""
        self.volume = max(0.0, min(1.5, float(value)))
        if self.running and self.proc.stdin:
            try:
                self.proc.stdin.write(json.dumps({"volume": self.volume}) + "\n")
                self.proc.stdin.flush()
            except OSError:
                pass
        return {"ok": True, "volume": self.volume}

    def summary(self):
        lags = sorted(self.lags)
        return {"phrases": len(lags),
                "delay_median_s": lags[len(lags) // 2] if lags else None,
                "transcript": str(self.log_path) if self.log_path else None}

    def status(self, n=10):
        with self.lock:
            return {"running": self.running, **self.info, **self.summary(), "state": self.state.snapshot(),
                    "captions": list(self.captions)[-n:]}

SESSION = Session()

def devices():
    if not is_set_up():
        return {"devices": []}
    out = subprocess.run([str(venv_python()), "-m", "lt_engine", "--list-devices"], cwd=str(HERE),
                         env=child_env({"PYTHONPATH": str(HERE), "PYTHONUTF8": "1"}), capture_output=True, text=True,
                         encoding="utf-8", errors="replace", timeout=60).stdout
    return json.loads(out.strip().splitlines()[-1]) if out.strip() else {"devices": []}


def _forget_voice_samples():
    """Voice workers get each speaker's sample as a temporary file (lt-voice-*): remove them once the session ends."""
    import glob
    import shutil
    import tempfile
    for d in glob.glob(os.path.join(tempfile.gettempdir(), "lt-voice-*")):
        shutil.rmtree(d, ignore_errors=True)
