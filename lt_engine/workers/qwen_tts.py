"""Qwen3-TTS Base (Apache-2.0): the speaker's own voice in 10 languages. Env: transformers 4.57 (qwen-tts)."""
import sys

from lt_engine.worker import serve
from lt_engine.workers._common import device, dtype

REPO = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
NAMES = {"zh": "chinese", "en": "english", "ja": "japanese", "ko": "korean", "de": "german", "fr": "french",
         "ru": "russian", "pt": "portuguese", "es": "spanish", "it": "italian"}


def load():
    from lt_engine.workers._common import limit_threads
    limit_threads()
    from qwen_tts import Qwen3TTSModel
    dev = device()
    m = Qwen3TTSModel.from_pretrained(REPO, device_map=dev, dtype=dtype(dev))
    m.lt_prompt = None
    return m


def ref(m, audio, sr, text):
    m.lt_prompt = m.create_voice_clone_prompt(ref_audio=(audio, sr), ref_text=text, x_vector_only_mode=not text)


def say(m, text, lang):
    if m.lt_prompt is None:
        raise RuntimeError("no speaker sample yet")
    wavs, sr = m.generate_voice_clone(text=text, language=NAMES.get(lang, "auto"), voice_clone_prompt=m.lt_prompt)
    return wavs[0], sr


if __name__ == "__main__":
    serve(load, ref, say)
