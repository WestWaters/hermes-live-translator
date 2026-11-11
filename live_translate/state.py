"""The translator's states, made explicit: one model that the tray, the Hermes tools, /translate and the captions
page all read, so every surface (people, developers, the agent) sees the same truth.

Session        not_set_up -> setting_up -> idle -> starting -> live -> stopping -> idle
                                    \\-> error          \\-> error      \\-> error (engine exit)
While live, four independent facets describe what is happening right now:
  activity     waiting | hearing | translating | speaking
  language     detecting | <code>                (the speaker's language, once identified)
  voice        stock | cloned                    (speaking in the speaker's own voice yet?)
  pace         on_time | catching_up             (speeding up / condensing to keep up)
"""
from __future__ import annotations

import threading
import time

SESSION_STATES = ("not_set_up", "setting_up", "idle", "starting", "live", "stopping", "error")
TRANSITIONS = {
    "not_set_up": {"setting_up"},
    "setting_up": {"idle", "error"},
    "idle": {"starting", "setting_up"},
    "starting": {"live", "stopping", "error", "idle"},
    "live": {"stopping", "error", "idle"},
    "stopping": {"idle", "error"},
    "error": {"idle", "setting_up", "starting"},
}
MESSAGES = {
    "not_set_up": "Not set up yet: run setup once (downloads the speech models).",
    "setting_up": "Setting up: installing the engine and downloading models.",
    "idle": "Ready. Pick your languages and press Start.",
    "starting": "Loading speech models (about 20-60 seconds)...",
    "live": "Live: listening and translating.",
    "stopping": "Finishing the last phrases...",
    "error": "Stopped:",
}


class IllegalTransition(RuntimeError):
    pass


class State:
    def __init__(self, initial="idle"):
        self._lock = threading.Lock()
        self.session = initial
        self.reason = ""
        self.since = time.time()
        self.facets = {}
        self.history = []
        self.version = 0
        self._changed = threading.Condition(self._lock)
        self._reset_facets()

    def _reset_facets(self):
        self.facets = {"activity": "waiting", "language": "detecting", "voice": "stock", "pace": "on_time"}

    def to(self, new, reason=""):
        with self._lock:
            if new == self.session:
                return
            if new not in TRANSITIONS.get(self.session, ()):
                raise IllegalTransition(f"{self.session} -> {new}")
            self.history.append((time.time(), self.session, new, reason))
            del self.history[:-50]
            self.session, self.reason, self.since = new, reason, time.time()
            if new in ("starting", "idle"):
                self._reset_facets()
            self.version += 1
            self._changed.notify_all()

    def can(self, new):
        return new in TRANSITIONS.get(self.session, ())

    def facet(self, **kw):
        with self._lock:
            changed = {k: v for k, v in kw.items() if self.facets.get(k) != v}
            if changed:
                self.facets.update(changed)
                self.version += 1
                self._changed.notify_all()

    def wait_change(self, version, timeout=15.0):
        """Block until the state moves past `version` (for live pages); returns the new version."""
        with self._lock:
            self._changed.wait_for(lambda: self.version != version, timeout=timeout)
            return self.version

    def snapshot(self):
        with self._lock:
            return {"session": self.session, "message": MESSAGES[self.session] + (f" {self.reason}" if self.reason
                                                                                    else ""),
                    "since": round(self.since, 1), "live": dict(self.facets) if self.session == "live" else None,
                    "can": sorted(TRANSITIONS.get(self.session, ())), "version": self.version}
