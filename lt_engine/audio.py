"""Audio in and out: files (any audio/video format) and microphones in, speakers or a virtual microphone out."""
from __future__ import annotations

import queue
import threading
import time

import numpy as np

SR = 16000
FRAME = 512                      # silero VAD frame (32 ms)


# ---------------------------------------------------------------- sources and sinks
def _decode(path):
    """Any audio or video file -> mono float32: soundfile for audio, ffmpeg for everything else (mp4, mkv, ...)."""
    import soundfile as sf
    try:
        a, sr = sf.read(path, dtype="float32")
        return (a.mean(1) if a.ndim > 1 else a), sr
    except Exception:
        import shutil
        import subprocess
        if not shutil.which("ffmpeg"):
            raise RuntimeError(f"cannot read {path}: install ffmpeg for video files")
        raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(SR),
                              "-"], capture_output=True, check=True).stdout
        return np.frombuffer(raw, np.float32).copy(), SR


class FileSource:
    """Plays a file into the pipeline at real speed (or as fast as possible with realtime=False)."""
    def __init__(self, path, realtime=True):
        a, sr = _decode(path)
        if sr != SR:
            a = np.interp(np.linspace(0, len(a) - 1, int(len(a) * SR / sr)), np.arange(len(a)), a)
        self.audio = np.concatenate([a.astype(np.float32), np.zeros(SR * 2, np.float32)])
        self.realtime = realtime
        self.q = queue.Queue()

    def start(self, t0):
        def run():
            for i in range(0, len(self.audio) - FRAME + 1, FRAME):
                if self.realtime:
                    d = t0 + i / SR - time.time()
                    if d > 0:
                        time.sleep(d)
                self.q.put(self.audio[i:i + FRAME])
            self.q.put(None)
        threading.Thread(target=run, daemon=True).start()


class MicSource:
    def __init__(self, device=None):
        self.device = device
        self.q = queue.Queue()

    def start(self, t0):
        import sounddevice as sd
        self.stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=FRAME,
                                     device=self.device, callback=lambda d, *_: self.q.put(d[:, 0].copy()))
        self.stream.start()


_PRIVATE = ("blackhole", "cable", "vb-audio", "loopback", "virtual", "soundflower", "monitor", "headphone",
            "headset", "airpods", "earbuds", "buds", "earphone", "beats", "bose qc")


def _audible_in_room(device):
    """True when translated speech comes out of real speakers (a microphone could hear it); False for headphones and
    virtual devices. Judged from the device's name; unknown devices count as speakers (the safe side)."""
    try:
        import sounddevice as sd
        name = sd.query_devices(device if device is not None else sd.default.device[1], "output")["name"].lower()
    except Exception:
        return True
    return not any(p in name for p in _PRIVATE)


class Sink:
    """Plays translated speech (default output or a virtual mic such as BlackHole) and/or keeps a timeline
    recording: original audio + translation at the moment the listener heard it."""
    def __init__(self, out_sr, device=None, play=True, record=None):
        self.sr, self.record = out_sr, record
        self.gain = 1.0                                    # live volume (0..1.5), set from the control line
        self.lock = threading.Lock()
        self.buf = np.zeros(0, np.float32)
        self.timeline = []
        self.play_end = 0.0
        self.stream = None
        self.on_play = None
        self.audible = play and _audible_in_room(device)    # its sound can reach a microphone in the room
        if play:
            import sounddevice as sd
            self.stream = sd.OutputStream(samplerate=out_sr, channels=1, dtype="float32", device=device,
                                          blocksize=1024, callback=self._cb)
            self.stream.start()

    def _cb(self, out, frames, *_):
        with self.lock:
            n = min(frames, len(self.buf))
            out[:n, 0] = np.clip(self.buf[:n] * self.gain, -1, 1)
            out[n:, 0] = 0
            self.buf = self.buf[n:]
        if self.on_play is not None:                       # the echo canceller hears exactly what we played
            self.on_play(out[:, 0])

    def push(self, chunk, now):
        """Queue a chunk; returns the stream time it starts playing. When the translation is running behind the
        speaker it is spoken up to 30% faster (same pitch) until it catches up."""
        from .stretch import speed_up
        backlog = self.play_end - now
        if backlog > 1.5:
            chunk = speed_up(chunk, 1.0 + min(0.3, (backlog - 1.5) * 0.1), self.sr)
        start = max(now, self.play_end)
        self.play_end = start + len(chunk) / self.sr
        if self.stream is not None:
            with self.lock:
                self.buf = np.concatenate([self.buf, chunk])
        self.timeline.append((start, chunk))
        return start

    def save(self, path, original=None, length_s=None):
        import soundfile as sf
        end = max([s + len(c) / self.sr for s, c in self.timeline] + [length_s or 0])
        mix = np.zeros(int(end * self.sr) + 1, np.float32)
        if original is not None:
            o = np.interp(np.linspace(0, len(original) - 1, int(len(original) * self.sr / SR)),
                          np.arange(len(original)), original).astype(np.float32) * 0.35
            mix[:len(o)] += o[:len(mix)]
        for s, c in self.timeline:
            i = int(s * self.sr)
            mix[i:i + len(c)] += c[:len(mix) - i]
        sf.write(path, np.clip(mix, -1, 1), self.sr)


def check_input(device):
    """Fail in a second, before loading any model, when there is no microphone to listen to."""
    import sounddevice as sd
    try:
        info = sd.query_devices(device, "input")
        return None if info and info.get("max_input_channels", 0) > 0 else "that input has no microphone channels"
    except Exception:
        names = [d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0]
        if device is None:
            return "no microphone found: plug one in" + (f", or pick an input ({', '.join(names[:3])})" if names else "")
        return f"input '{device}' not found" + (f"; available: {', '.join(names[:4])}" if names else "")
