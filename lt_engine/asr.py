"""Speech recognition with two engines and one rule: fast where possible, every language where needed.

Qwen3-ASR 0.6B (30 languages + 22 Chinese dialects, ~0.15 s per phrase on an M4) is the fast path. Whisper
large-v3-turbo (99 languages) identifies each new speaker's language once and transcribes every language Qwen3-ASR
does not know.

Phrases are cut at pauses by voice activity, so a phrase is normally transcribed once. Long run-on speech is
re-transcribed every STEP seconds and the prefix two passes agree on is committed early (LocalAgreement-2):
committed text is never taken back."""
from __future__ import annotations

import re

import numpy as np

SR = 16000

QWEN_LANGS = {
    "Chinese": "zh", "English": "en", "Cantonese": "yue", "Arabic": "ar", "German": "de", "French": "fr",
    "Spanish": "es", "Portuguese": "pt", "Indonesian": "id", "Italian": "it", "Korean": "ko", "Russian": "ru",
    "Thai": "th", "Vietnamese": "vi", "Japanese": "ja", "Turkish": "tr", "Hindi": "hi", "Malay": "ms",
    "Dutch": "nl", "Swedish": "sv", "Danish": "da", "Finnish": "fi", "Polish": "pl", "Czech": "cs",
    "Filipino": "tl", "Persian": "fa", "Greek": "el", "Romanian": "ro", "Hungarian": "hu", "Macedonian": "mk",
}
QWEN_BY_CODE = {v: k for k, v in QWEN_LANGS.items()}

ID_MIN_S = 3.0                       # speech needed to tell the language reliably (measured: 1.5 s fails)
ID_MAX_S = 7.0                       # if the two identifiers still disagree, decide by then
# Whisper writes Chinese in either script; a Simplified sentence as context keeps it in Simplified (and punctuated).
PROMPTS = {"zh": "以下是普通话的句子。"}
RECHECK_S = 2.0           # each phrase this long is checked once for a change of language
SWITCH_P = 0.8            # ...and switches only on an identification this confident
SAME_SPEECH = {"hi": ("ur",), "ur": ("hi",), "zh": ("yue",), "yue": ("zh",)}   # spoken alike: always ask both
QWEN_MODEL = "mlx-community/Qwen3-ASR-0.6B-8bit"
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"


class Qwen:
    def __init__(self, model=QWEN_MODEL):
        from mlx_audio.stt.utils import load_model
        from .pins import local
        self.m = load_model(local(model))

    def transcribe(self, audio, lang=None):
        import mlx.core as mx
        r = self.m.generate(mx.array(audio.astype(np.float32)), language=QWEN_BY_CODE.get(lang) if lang else None)
        detected = r.language[0] if isinstance(r.language, list) and r.language else r.language
        return r.text.strip(), QWEN_LANGS.get(detected, lang)


class Whisper:
    def __init__(self, model=WHISPER_MODEL):
        import mlx_whisper  # noqa: F401
        self.model = model

    def transcribe(self, audio, lang=None):
        import mlx_whisper
        from .pins import local
        r = mlx_whisper.transcribe(audio.astype(np.float32), path_or_hf_repo=local(self.model), language=lang,
                                   temperature=0.0, condition_on_previous_text=False, verbose=None,
                                   initial_prompt=PROMPTS.get(lang))
        return r["text"].strip(), r.get("language", lang)

    def detect(self, audio):
        """(language, probability) of `audio` by Whisper's 99-language identifier."""
        import mlx.core as mx
        from mlx_whisper.audio import N_SAMPLES, log_mel_spectrogram, pad_or_trim
        from mlx_whisper.load_models import load_model
        if not hasattr(self, "_m"):
            from .pins import local
            self._m = load_model(local(self.model))
        mel = log_mel_spectrogram(pad_or_trim(mx.array(audio.astype(np.float32)), N_SAMPLES),
                                  n_mels=self._m.dims.n_mels)
        _, probs = self._m.detect_language(mel)
        lang = max(probs, key=probs.get)
        return lang, float(probs[lang])


class FasterWhisper:
    """Whisper on CTranslate2 (Windows / Linux, NVIDIA or CPU): 99 languages with identification."""

    def __init__(self, model=None):
        from faster_whisper import WhisperModel
        from .hw import detect
        gpu = detect() == "cuda"
        import os
        threads = int(os.environ.get("LIVE_TRANSLATE_THREADS", "0") or 0)
        # CPU: "small" -- measured on a desktop CPU it writes the same text as turbo 4x faster (1.25 s vs 5.1 s
        # for a 4 s phrase) with the language still identified; "base" is faster again but makes mistakes.
        from .pins import local                             # pinned CTranslate2 conversions, by repo
        name = model or ("dropbox-dash/faster-whisper-large-v3-turbo" if gpu else "Systran/faster-whisper-small")
        self.m = WhisperModel(local(name) if "/" in name else name, device="cuda" if gpu else "cpu",
                              compute_type="float16" if gpu else "int8", cpu_threads=threads)

    def transcribe(self, audio, lang=None):
        segs, info = self.m.transcribe(audio.astype(np.float32), language=lang, beam_size=1,
                                       condition_on_previous_text=False, initial_prompt=PROMPTS.get(lang))
        return "".join(s.text for s in segs).strip(), info.language

    def detect(self, audio):
        _, info = self.m.transcribe(audio.astype(np.float32), beam_size=1, without_timestamps=True)
        return info.language, float(info.language_probability)


class TorchWhisper:
    """Whisper large-v3-turbo on PyTorch (transformers), for machines where CTranslate2 has no build -- Windows on
    Arm with an NVIDIA GPU (RTX Spark). Same model, same transcribe/detect contract as FasterWhisper."""
    MODEL = "openai/whisper-large-v3-turbo"

    def __init__(self, model=None):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.dev == "cuda" else torch.float32
        from .pins import local
        path = local(model or self.MODEL)
        self.p = WhisperProcessor.from_pretrained(path)
        self.m = WhisperForConditionalGeneration.from_pretrained(path, dtype=self.dtype).to(self.dev).eval()
        gc = self.m.generation_config
        self.langs = {tok.strip("<|>"): i for tok, i in gc.lang_to_id.items()}

    def _feats(self, audio):
        f = self.p(audio.astype(np.float32), sampling_rate=16000, return_tensors="pt").input_features
        return f.to(self.dev, self.dtype)

    def transcribe(self, audio, lang=None):
        feats = self._feats(audio)
        if lang is None:
            lang, _ = self._detect(feats)
        kw = {"prompt_ids": self.torch.tensor(self.p.get_prompt_ids(PROMPTS[lang]), device=self.dev)} if lang in PROMPTS else {}
        with self.torch.inference_mode():
            ids = self.m.generate(feats, language=lang, task="transcribe", num_beams=1, **kw)
        text = self.p.batch_decode(ids, skip_special_tokens=True)[0]
        if kw:                                              # the prompt is echoed at the start of the output
            text = text.replace(PROMPTS[lang], "", 1)
        return text.strip(), lang

    def _detect(self, feats):
        sot = self.m.generation_config.decoder_start_token_id
        with self.torch.inference_mode():
            logits = self.m(input_features=feats, decoder_input_ids=self.torch.tensor([[sot]], device=self.dev)).logits[0, -1]
        codes = list(self.langs)
        probs = self.torch.softmax(logits[[self.langs[c] for c in codes]].float(), dim=-1)
        i = int(probs.argmax())
        return codes[i], float(probs[i])

    def detect(self, audio):
        """(language, probability) by Whisper's own language identifier."""
        return self._detect(self._feats(audio))


def _faster_whisper_or_torch(model):
    try:
        import faster_whisper  # noqa: F401
        return FasterWhisper(model)
    except ImportError:                                     # no CTranslate2 build here (Windows on Arm)
        return TorchWhisper()


class MLXAudioSTT:
    """Any mlx-audio speech-to-text model (Nemotron, Parakeet, Voxtral, Canary, ...)."""

    def __init__(self, repo):
        from mlx_audio.stt.utils import load_model
        from .pins import local
        self.m = load_model(local(repo))

    def transcribe(self, audio, lang=None):
        import inspect
        import mlx.core as mx
        kw = {"language": lang} if lang and "language" in inspect.signature(self.m.generate).parameters else {}
        r = self.m.generate(mx.array(audio.astype(np.float32)), **kw)
        return (getattr(r, "text", "") or "").strip(), lang


class OpenAISTT:
    """Any OpenAI-compatible /audio/transcriptions server, local or a subscription (key: LIVE_TRANSLATE_STT_KEY)."""

    def __init__(self, base, model):
        self.url, self.model = base.rstrip("/") + "/audio/transcriptions", model or "whisper-1"

    def transcribe(self, audio, lang=None):
        import io
        import json
        import os
        import urllib.request
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, audio.astype(np.float32), SR, format="WAV")
        b = "----lt" + os.urandom(8).hex()
        fields = {"model": self.model, **({"language": lang} if lang else {})}
        body = b"".join(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
                        for k, v in fields.items())
        body += (f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
                 f"Content-Type: audio/wav\r\n\r\n").encode() + buf.getvalue() + f"\r\n--{b}--\r\n".encode()
        key = os.environ.get("LIVE_TRANSLATE_STT_KEY", "")
        req = urllib.request.Request(self.url, body, {"Content-Type": f"multipart/form-data; boundary={b}",
                                                      **({"Authorization": f"Bearer {key}"} if key else {})})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r).get("text", "").strip(), lang


# Recognition routes, like voice routes: "zh,ja=<engine>; *=<engine>". Engines: qwen3-asr[:repo], whisper[:repo],
# mlx-audio:<repo>, openai:<base url>#<model>.
ASR_PRESETS = {"auto-cuda": "*=faster-whisper", "auto": "zh,en,yue,ar,de,fr,es,pt,id,it,ko,ru,th,vi,ja,tr,hi,ms,nl,sv,da,fi,pl,cs,tl,fa,el,ro,hu,mk="
                       "qwen3-asr; *=whisper",
               "whisper": "*=whisper"}


def make_stt(spec):
    kind, _, rest = spec.strip().partition(":")
    if kind == "qwen3-asr":
        return Qwen(rest or QWEN_MODEL)
    if kind == "whisper":
        return Whisper(rest or WHISPER_MODEL)
    if kind == "faster-whisper":
        return _faster_whisper_or_torch(rest or None)
    if kind == "mlx-audio":
        return MLXAudioSTT(rest)
    if kind == "openai":
        base, _, model = rest.partition("#")
        return OpenAISTT(base, model)
    raise ValueError(f"unknown recognition engine: {spec!r}")


def tokens(text, no_space):
    return list(text) if no_space else re.findall(r"\S+\s*", text)


class Recognizer:
    """One speaker stream. Call `utterance()` at the start of speech, `partial()` during long speech, `final()` at
    a pause; each returns only the NEW committed text."""

    def __init__(self, language=None, fast=True, route="auto", shared=None):
        from . import langs
        self.langs = langs
        from .hw import detect
        mlx = detect() == "mlx"
        if route == "auto" and not mlx:
            route = "auto-cuda"                               # faster-whisper on NVIDIA / CPU
        route = ASR_PRESETS.get(route if (fast or not mlx) else "whisper", route)
        self.idspec = "whisper" if mlx else "faster-whisper"
        self.route = []
        for part in route.split(";"):
            if part.strip():
                ls, _, spec = part.partition("=")
                self.route.append(({x.strip() for x in ls.split(",")}, spec.strip()))
        self.made = shared if shared is not None else {}     # two-way lanes share the loaded models
        self.whisper = self._engine(self.idspec)              # always: it identifies the language (99)
        qspec = next((sp for _, sp in self.route if sp.startswith("qwen3-asr")), None)
        self.qwen = self._engine(qspec) if qspec else None
        self.fixed = language
        self.language = language
        self.engine = None
        self._pick()
        self.committed = ""
        self.prev = None
        self.held = []                                        # short first phrases waiting for identification

    def _engine(self, spec):
        if spec in ("whisper", "faster-whisper") and spec in self.made:
            return self.made[spec]
        if spec not in self.made:
            self.made[spec] = make_stt(spec)
        return self.made[spec]

    def _pick(self):
        self.engine = None
        if self.language is None:
            return
        for ls, spec in self.route:
            if self.language in ls or "*" in ls:
                self.engine = self._engine(spec)
                return
        self.engine = self.whisper

    def warm(self):
        z = np.zeros(SR, np.float32)
        self.whisper.transcribe(z, "en")                     # loads the model
        if self.qwen is not None:
            self.qwen.transcribe(z)

    def utterance(self, pause_s=0.0):
        self.committed, self.prev = "", None
        self.checked = False
        if pause_s > 3.0 and self.fixed is None:          # maybe a different speaker: identify again
            self.language = None
            self._pick()

    def _identify(self, audio):
        """Pick the speaker's language from their speech so far (>= ID_MIN_S) by two independent votes, so
        background voices in another language cannot hijack it: Whisper's identifier (99 languages) and
        Qwen3-ASR's (98.7% on FLEURS, its 30). Agreement locks it; on disagreement Qwen3-ASR wins inside its 30,
        Whisper only when very sure; otherwise wait for more speech (up to ID_MAX_S). Returns False to wait."""
        wl, wp = self.whisper.detect(audio)
        ql = None
        if self.qwen is not None:
            _, ql = self.qwen.transcribe(audio, None)
            ql = ql if ql in QWEN_BY_CODE else None
        long_enough = len(audio) >= SR * ID_MAX_S
        if ql and ql == wl:                                   # both agree
            choice = ql
        elif ql and wl in SAME_SPEECH.get(ql, ()):            # Hindi/Urdu, Mandarin/Cantonese: Qwen3-ASR decides
            choice = ql
        elif ql and wl in QWEN_BY_CODE:                       # both inside Qwen3-ASR's 30: it is the better judge
            choice = ql
        elif ql:                                              # Whisper hears a language Qwen3-ASR does not know
            choice = wl if wp >= 0.9 else (ql if long_enough else None)
        else:                                                 # Qwen3-ASR has no opinion (or is not installed)
            choice = wl if (wp >= 0.8 or long_enough or self.qwen is None) else None
        if choice is None:
            return False                                      # not sure yet: hear more first
        self.language = choice
        self._pick()
        return True

    def _run(self, audio, final=False):
        if self.language is None:
            seen = np.concatenate(self.held + [audio]) if self.held else audio
            if len(seen) < SR * ID_MIN_S or (self.held and not final):
                if final:                                     # too little speech to tell: hold it for a moment
                    self.held.append(audio)
                return ""
            if not self._identify(seen):                     # undecided: keep holding this speech
                if final:
                    self.held.append(audio)
                return ""
            self.checked = True
            if self.held:                                     # the held phrase, now in the right language
                prefix = self.engine.transcribe(np.concatenate(self.held), self.language)[0]
                self.held = []
                return (prefix + (" " if self.language not in self.langs.NO_SPACE else "") +
                        self.engine.transcribe(audio, self.language)[0]).strip()
        elif (self.fixed is None and not getattr(self, "checked", True) and self.committed == ""
              and len(audio) >= SR * RECHECK_S):
            # Once per phrase (its first 2+ s, before anything is committed): is this still the same language?
            # People switch languages, and a new speaker may not be told apart by voice. Only a confident
            # identification switches -- short or noisy phrases keep the language they have.
            self.checked = True
            wl, wp = self.whisper.detect(audio)
            self.last_check = (wl, round(wp, 2))
            if wl != self.language and (wp >= SWITCH_P or wl in SAME_SPEECH.get(self.language, ())):
                # a likely change: confirm it with the full two-detector vote (Qwen3-ASR settles Hindi vs Urdu,
                # Chinese vs Cantonese, which Whisper confuses); keep the language if the vote is unsure
                old = self.language
                if not self._identify(audio):
                    self.language = old
                self._pick()
        return self.engine.transcribe(audio, self.language)[0]

    def partial(self, audio):
        text = self._run(audio)
        ns = self.language in self.langs.NO_SPACE
        cur = tokens(text, ns)
        if self.prev is not None:
            n = 0
            while n < min(len(cur), len(self.prev)) and cur[n].strip() == self.prev[n].strip():
                n += 1
            agreed = "".join(cur[:n])
            if len(agreed) > len(self.committed) and agreed.startswith(self.committed):
                new = agreed[len(self.committed):]
                self.committed = agreed
                self.prev = cur
                return new
        self.prev = cur
        return ""

    def final(self, audio):
        text = self._run(audio, final=True)
        if self.committed and text.startswith(self.committed):
            new = text[len(self.committed):]
        elif self.committed:                                  # the last pass rephrased: keep what is left after
            new = text[len(self.committed):] if len(text) > len(self.committed) else ""
        else:
            new = text
        self.committed, self.prev = "", None
        return new
