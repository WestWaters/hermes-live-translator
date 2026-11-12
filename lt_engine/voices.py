"""Voice backends: any text-to-speech the user wants, local or a subscription, routed per language.

A voice spec is one of:
  mlx-audio:<hf repo>[?clone=0&steps=16]
                                    any model mlx-audio can load (OmniVoice, VoxCPM2, Qwen3-TTS, Fish Drama 3,
                                    Higgs 3, MOSS-TTS, Chatterbox, ... and whatever it adds later); clones the speaker
  openai:<base url>#<model>#<voice> any OpenAI-compatible /audio/speech server, local or cloud (key:
                                    LIVE_TRANSLATE_TTS_KEY)
  elevenlabs:<model>                ElevenLabs with the speaker's voice cloned once per session
                                    (key: ELEVENLABS_API_KEY)
  command:<shell template>          any program: {text} {lang} {ref_wav} {ref_text} {out_wav} are filled in
  worker:<command line>             any voice model in its own process/environment (lt_engine/worker.py protocol)
  env:<name>[ args]                 a bundled worker in the environment setup created: qwen-tts, omnivoice,
                                    chatterbox (Windows / Linux / NVIDIA or CPU)
  system                            the operating system's voice (last resort)

A route is "zh,ja,ko=<spec>; *=<spec>": the first entry that lists the target language (or *) speaks it.
"""
from __future__ import annotations

import atexit
import json
import os
import shlex
import subprocess
import sys
import tempfile
import urllib.request

import numpy as np

OUT_SR = 24000

# Presets. "best": the speaker's own voice in 646 languages (OmniVoice weights are CC-BY-NC: personal and research
# use). "commercial": Apache/MIT models only -- own voice in ~40 languages, then the system voice.
# Measured on real speakers (FLEURS): Qwen3-TTS 1.7B and Chatterbox clone across languages; OmniVoice's MLX port
# clones reliably only within a language, so it speaks the long tail in a natural voice matched to the speaker.
PRESETS = {
    "best": "zh,en,ja,ko,de,fr,ru,pt,es,it=mlx-audio:mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit; "
            "ar,da,el,fi,he,hi,ms,nl,no,pl,sv,sw,tr=mlx-audio:mlx-community/chatterbox-4bit; "
            "*=mlx-audio:mlx-community/OmniVoice-bf16?clone=0&steps=16",
    # NVIDIA GPU or CPU (Windows / Linux): the same model families, each in its own environment.
    "best-cuda": "zh,en,ja,ko,de,fr,ru,pt,es,it=env:qwen-tts; "
                 "ar,da,el,fi,he,hi,ms,nl,no,pl,sv,sw,tr=env:chatterbox; *=env:omnivoice",
    "commercial-cuda": "zh,en,ja,ko,de,fr,ru,pt,es,it=env:qwen-tts; "
                       "ar,da,el,fi,he,hi,ms,nl,no,pl,sv,sw,tr=env:chatterbox; *=system",
    # CPU only: voice cloning cannot keep up live on a CPU (Qwen3-TTS 0.6B took 11-59 s a sentence on a desktop
    # CPU), so Kokoro's natural voices (82M, ~0.5x real time) speak its 8 languages, matched to the speaker's
    # pitch; the cloning models remain for the others, and a GPU brings cloning back.
    "best-cpu": "en,es,fr,hi,it,pt,ja,zh=kokoro; ko,de,ru=env:qwen-tts Qwen/Qwen3-TTS-12Hz-0.6B-Base; "
                "ar,da,el,fi,he,ms,nl,no,pl,sv,sw,tr=env:chatterbox; *=env:omnivoice",
    "commercial-cpu": "en,es,fr,hi,it,pt,ja,zh=kokoro; ko,de,ru=env:qwen-tts Qwen/Qwen3-TTS-12Hz-0.6B-Base; "
                      "ar,da,el,fi,he,ms,nl,no,pl,sv,sw,tr=env:chatterbox; *=system",
    "commercial": "zh,en,ja,ko,de,fr,ru,pt,es,it=mlx-audio:mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit; "
                  "ar,da,el,fi,he,hi,ms,nl,no,pl,sv,sw,tr=mlx-audio:mlx-community/chatterbox-4bit; "
                  "*=system",
}
LICENCES = {"OmniVoice": "CC-BY-NC-4.0 (non-commercial)", "Qwen3-TTS": "Apache-2.0", "chatterbox": "MIT",
            "VoxCPM2": "Apache-2.0", "kokoro": "Apache-2.0 (reads text with espeak-ng, GPL-3.0)", "MOSS-TTS": "Apache-2.0", "higgs": "non-commercial", "drama": "Fish Community"}

QWEN_NAMES = {"zh": "chinese", "en": "english", "ja": "japanese", "ko": "korean", "de": "german", "fr": "french",
              "ru": "russian", "pt": "portuguese", "es": "spanish", "it": "italian"}


def _resample(a, sr, to=OUT_SR):
    if sr == to or len(a) == 0:
        return np.asarray(a, np.float32)
    n = int(round(len(a) * to / sr))
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)


def _read_wav(path):
    import soundfile as sf
    a, sr = sf.read(path, dtype="float32")
    return _resample(a.mean(1) if a.ndim > 1 else a, sr)


class Voice:
    name = "voice"

    def warm(self, lang):
        pass

    def set_reference(self, audio16k, text):
        self.ref = (audio16k, text)

    def speak(self, text, lang):
        raise NotImplementedError


def speaker_gender(audio16k):
    """'male' or 'female' from the median pitch of voiced frames (autocorrelation); used to match a natural voice."""
    a = np.asarray(audio16k, np.float32)
    f0s = []
    for i in range(0, len(a) - 1024, 1024):
        x = a[i:i + 1024] - a[i:i + 1024].mean()
        if np.sqrt((x ** 2).mean()) < 0.01:
            continue
        ac = np.correlate(x, x, "full")[1023:]
        lo, hi = 16000 // 400, 16000 // 60                    # 60-400 Hz
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > 0.3 * ac[0]:
            f0s.append(16000 / k)
    return "female" if f0s and float(np.median(f0s)) >= 165 else "male"


_LOADED = {}


class MLXAudioVoice(Voice):
    """Any mlx-audio TTS. Model families differ only in how they take the language and the reference voice."""

    def __init__(self, repo, clone=True, steps=None):
        self.repo, self.m, self.ref, self.conds = repo, None, None, None
        self.clone, self.steps, self.gender = clone, steps, None
        self.name = repo.split("/")[-1] + ("" if clone else " (natural voice)")

    def _model(self):
        if self.m is None:
            if self.repo not in _LOADED:                           # one copy per model, shared by both lanes
                from mlx_audio.tts.utils import load_model
                from .pins import local
                _LOADED[self.repo] = load_model(local(self.repo))
            self.m = _LOADED[self.repo]
            self.kind = type(self.m).__module__.split(".")[-2]      # e.g. qwen3_tts, chatterbox, omnivoice
        return self.m

    def warm(self, lang):
        import mlx.core as mx
        # Warm with a reference of the same length the engine uses (5 s), so the first real phrase reuses it.
        noise = mx.array((np.random.default_rng(0).standard_normal(16000 * 5) * 0.01).astype(np.float32))
        saved, self.ref = self.ref, (noise, "ok") if self.clone else None
        try:
            for _ in self.speak("Hello, this is a short warm-up sentence.", lang):
                pass
        finally:
            self.ref, self.conds = saved, None

    def set_reference(self, audio16k, text):
        import mlx.core as mx
        self.gender = speaker_gender(audio16k)
        self.ref = (mx.array(np.asarray(audio16k, np.float32)), text) if self.clone else None
        self.conds = None

    def _kwargs(self, lang):
        m = self._model()
        kw = {"verbose": False}
        if self.kind == "qwen3_tts":
            kw["lang_code"] = QWEN_NAMES.get(lang, "auto")
        elif self.kind == "chatterbox":
            kw["lang_code"] = lang
        else:
            kw["lang_code"] = lang
            kw["language"] = lang
        if self.steps:
            kw["num_steps"] = self.steps
        if not self.clone and self.gender:
            kw["instruct"] = self.gender
        if self.ref is not None:
            if self.kind == "chatterbox":
                if self.conds is None:
                    self.conds = m.prepare_conditionals(self.ref[0], 16000, exaggeration=0.3)
                kw["conds"] = self.conds
            else:
                kw["ref_audio"] = self.ref[0]
                if self.ref[1]:
                    kw["ref_text"] = self.ref[1]
        return kw

    def speak(self, text, lang):
        import inspect
        m = self._model()
        kw = self._kwargs(lang)
        params = inspect.signature(m.generate).parameters
        accepts_any = any(p.kind == p.VAR_KEYWORD for p in params.values())
        kw = {k: v for k, v in kw.items() if accepts_any or k in params}
        if "stream" in params or accepts_any:
            kw.update(stream=True, streaming_interval=0.4)
        for r in m.generate(text=text, **kw):
            yield _resample(np.asarray(r.audio, np.float32).reshape(-1), m.sample_rate)


class OpenAIVoice(Voice):
    def __init__(self, base, model, voice):
        self.url, self.model, self.voice = base.rstrip("/") + "/audio/speech", model, voice or "alloy"
        self.name = f"openai:{model}"
        self.ref = None

    def speak(self, text, lang):
        key = os.environ.get("LIVE_TRANSLATE_TTS_KEY", "")
        body = json.dumps({"model": self.model, "input": text, "voice": self.voice, "response_format": "wav"}).encode()
        req = urllib.request.Request(self.url, body, {"Content-Type": "application/json",
                                                      **({"Authorization": f"Bearer {key}"} if key else {})})
        with urllib.request.urlopen(req, timeout=60) as r, tempfile.NamedTemporaryFile(suffix=".wav") as f:
            f.write(r.read())
            f.flush()
            yield _read_wav(f.name)


class ElevenLabsVoice(Voice):
    """The speaker's voice is cloned once (instant voice clone) and used for every phrase."""
    API = "https://api.elevenlabs.io/v1"

    def __init__(self, model="eleven_flash_v2_5"):
        self.model, self.voice_id, self.ref = model or "eleven_flash_v2_5", None, None
        self.name = f"elevenlabs:{self.model}"
        _OPEN.append(self)                                  # deleted from the account when the engine stops

    def close(self):
        """Delete the voice this session cloned in the user's ElevenLabs account (they are permanent otherwise)."""
        vid, self.voice_id = self.voice_id, None
        if not vid:
            return
        try:
            req = urllib.request.Request(f"{self.API}/voices/{vid}", method="DELETE", headers={"xi-api-key": self._key()})
            urllib.request.urlopen(req, timeout=20).close()
        except Exception as e:                              # never block shutdown; say what is left behind
            sys.stderr.write(f"elevenlabs: could not delete cloned voice {vid}: {e}\n")

    def _key(self):
        k = os.environ.get("ELEVENLABS_API_KEY")
        if not k:
            raise RuntimeError("ELEVENLABS_API_KEY is not set")
        return k

    def set_reference(self, audio16k, text):
        import io
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, np.asarray(audio16k, np.float32), 16000, format="WAV")
        boundary = "----lt" + os.urandom(8).hex()
        parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="name"\r\n\r\nlive-translate speaker\r\n',
                 f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="ref.wav"\r\n'
                 f'Content-Type: audio/wav\r\n\r\n']
        body = "".join(parts).encode() + buf.getvalue() + f"\r\n--{boundary}--\r\n".encode()
        self.close()                                        # a new speaker replaces the previous clone
        req = urllib.request.Request(f"{self.API}/voices/add", body,
                                     {"xi-api-key": self._key(),
                                      "Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.voice_id = json.load(r)["voice_id"]

    def speak(self, text, lang):
        vid = self.voice_id or "21m00Tcm4TlvDq8ikWAM"
        body = json.dumps({"text": text, "model_id": self.model}).encode()
        req = urllib.request.Request(f"{self.API}/text-to-speech/{vid}/stream?output_format=pcm_24000", body,
                                     {"xi-api-key": self._key(), "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            while True:
                chunk = r.read(9600)
                if not chunk:
                    return
                yield np.frombuffer(chunk[: len(chunk) // 2 * 2], np.int16).astype(np.float32) / 32768.0


_OPEN = []                                                  # cloud voices this engine created


def close_all():
    """Clean up anything a voice created outside this computer (ElevenLabs clones). Safe to call twice."""
    while _OPEN:
        try:
            _OPEN.pop().close()
        except Exception:
            pass


atexit.register(close_all)


class CommandVoice(Voice):
    def __init__(self, template):
        self.template, self.ref, self.name = template, None, "command"

    def speak(self, text, lang):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as d:
            ref_wav = os.path.join(d, "ref.wav")
            if self.ref is not None:
                sf.write(ref_wav, np.asarray(self.ref[0], np.float32), 16000)
            out = os.path.join(d, "out.wav")
            cmd = self.template.format(text=shlex.quote(text), lang=lang, ref_wav=shlex.quote(ref_wav),
                                       ref_text=shlex.quote(self.ref[1] if self.ref else ""), out_wav=out)
            subprocess.run(cmd, shell=True, check=True, timeout=120)
            yield _read_wav(out)


class KokoroVoice(Voice):
    """Kokoro-82M (Apache-2.0) on ONNX Runtime: natural voices in 8 languages, fast on any CPU. No cloning: it picks
    a female or male voice to match the speaker. Speaks sentence by sentence so the first audio comes quickly."""
    name = "kokoro"
    REPO = "onnx-community/Kokoro-82M-v1.0-ONNX"
    VOICES = {"en": ("af_heart", "am_michael"), "es": ("ef_dora", "em_alex"), "fr": ("ff_siwis", "ff_siwis"),
              "hi": ("hf_alpha", "hm_omega"), "it": ("if_sara", "im_nicola"), "pt": ("pf_dora", "pm_alex"),
              "ja": ("jf_alpha", "jm_kumo"), "zh": ("zf_xiaobei", "zm_yunjian")}
    ESPEAK = {"en": "en-us", "es": "es", "fr": "fr-fr", "hi": "hi", "it": "it", "pt": "pt-br", "ja": "ja",
              "zh": "cmn"}

    def __init__(self):
        self.ref, self.gender, self.k = None, "female", None

    def _model(self):
        if self.k is None:
            import onnxruntime as ort
            from kokoro_onnx import Kokoro
            from .pins import local
            model = local(self.REPO, "onnx/model_quantized.onnx")
            npz = os.path.join(os.path.dirname(model), "lt-voices.npz")
            if not os.path.exists(npz):                           # the per-voice files, packed as kokoro-onnx wants
                names = sorted({v for pair in self.VOICES.values() for v in pair})
                np.savez(npz, **{v: np.fromfile(local(self.REPO, f"voices/{v}.bin"), np.float32)
                                 .reshape(-1, 1, 256) for v in names})
            so = ort.SessionOptions()
            so.intra_op_num_threads = int(os.environ.get("LIVE_TRANSLATE_THREADS", "0") or 0)
            self.k = Kokoro.from_session(ort.InferenceSession(model, so, providers=["CPUExecutionProvider"]), npz)
        return self.k

    def warm(self, lang):
        for _ in self.speak("Hello.", lang):
            pass

    def set_reference(self, audio16k, text):
        self.gender = speaker_gender(audio16k)

    def speak(self, text, lang):
        import re
        k = self._model()
        female, male = self.VOICES.get(lang, self.VOICES["en"])
        voice = male if self.gender == "male" else female
        for part in re.split(r"(?<=[.!?。！？;；])\s*", text):
            if part.strip():
                a, sr = k.create(part, voice=voice, speed=1.0, lang=self.ESPEAK.get(lang, "en-us"))
                yield _resample(np.asarray(a, np.float32), sr)


class SystemVoice(Voice):
    name = "system"

    def __init__(self):
        self.ref = None
        self.voices = {}
        if sys.platform == "darwin":
            try:
                out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=10).stdout
                for line in out.splitlines():
                    parts = line.split("#")[0].split()
                    if len(parts) >= 2:
                        self.voices.setdefault(parts[-1].split("_")[0].lower(), " ".join(parts[:-1]))
            except Exception:
                pass

    def speak(self, text, lang):
        import shutil
        with tempfile.TemporaryDirectory() as d:
            wav = os.path.join(d, "o.wav")
            if sys.platform == "darwin":
                cmd = ["say", "-o", wav, f"--data-format=LEF32@{OUT_SR}", "--file-format=WAVE"]
                if lang in self.voices:
                    cmd += ["-v", self.voices[lang]]
                subprocess.run(cmd + [text], check=True, timeout=60)
            elif os.name == "nt":                              # Windows' built-in speech (SAPI)
                ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                      f"$s.SetOutputToWaveFile('{wav}'); $s.Speak([Console]::In.ReadToEnd()); $s.Dispose()")
                subprocess.run(["powershell", "-NoProfile", "-Command", ps], input=text, text=True, check=True,
                               timeout=60)
            elif shutil.which("espeak-ng"):
                subprocess.run(["espeak-ng", "-v", lang, "-w", wav, text], check=True, timeout=60)
            else:
                return
            yield _read_wav(wav)


def make(spec):
    kind, _, rest = spec.strip().partition(":")
    if kind == "mlx-audio":
        from urllib.parse import parse_qs
        repo, _, q = rest.partition("?")
        o = {k: v[-1] for k, v in parse_qs(q).items()}
        return MLXAudioVoice(repo, clone=o.get("clone", "1") != "0", steps=int(o["steps"]) if "steps" in o else None)
    if kind == "openai":
        base, _, more = rest.partition("#")
        model, _, voice = more.partition("#")
        return OpenAIVoice(base, model, voice)
    if kind == "elevenlabs":
        return ElevenLabsVoice(rest)
    if kind == "command":
        return CommandVoice(rest)
    if kind == "worker":
        from .worker import WorkerVoice
        return WorkerVoice(rest)
    if kind == "env":
        from .worker import WorkerVoice
        name, _, args = rest.partition(" ")
        py = env_python(name)
        return WorkerVoice([py, "-m", f"lt_engine.workers.{name.replace('-', '_')}"] + args.split(), name=name)
    if kind == "system":
        return SystemVoice()
    if kind == "kokoro":
        return KokoroVoice()
    raise ValueError(f"unknown voice spec: {spec!r}")


def env_python(name):
    """The interpreter of a worker environment created by setup (<data>/envs/<name>)."""
    base = os.environ.get("LIVE_TRANSLATE_ENVS") or os.path.join(
        os.environ.get("HERMES_HOME") or os.path.join(os.path.expanduser("~"), ".hermes"),
        "plugin-data", "live-translate", "envs")
    exe = os.path.join(base, name, "Scripts", "python.exe") if os.name == "nt" else os.path.join(base, name, "bin",
                                                                                               "python")
    if not os.path.exists(exe):
        raise RuntimeError(f"voice environment '{name}' is not installed (run setup)")
    return exe


class Router:
    """Routes each target language to a voice; voices load on first use, so unused ones cost nothing."""

    def __init__(self, route="commercial"):
        from .hw import detect
        hw = detect()
        if route in ("best", "commercial") and hw != "mlx":
            route += "-cuda" if hw == "cuda" else "-cpu"  # same choice, PyTorch workers off the Mac
        route = PRESETS.get(route, route)
        self.entries, self.made = [], {}
        for part in route.split(";"):
            if part.strip():
                langs_, _, spec = part.partition("=")
                self.entries.append(({x.strip() for x in langs_.split(",")}, spec.strip()))
        self.ref = None

    def voice(self, lang):
        for langs_, spec in self.entries:
            if lang in langs_ or "*" in langs_:
                if spec not in self.made:
                    v = make(spec)
                    if self.ref is not None:
                        v.set_reference(*self.ref)
                    self.made[spec] = v
                return self.made[spec]
        return SystemVoice()

    def set_reference(self, audio16k, text):
        self.ref = (audio16k, text)
        for v in self.made.values():
            v.set_reference(audio16k, text)

    def speak(self, text, lang):
        yield from self.voice(lang).speak(text, lang)
