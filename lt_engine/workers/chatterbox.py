"""Chatterbox multilingual (MIT): the speaker's own voice in 23 languages, no transcript needed.
Env: chatterbox-tts installed with --no-deps on a current torch (its torch 2.6 pin does not run on RTX 50xx)."""
import os
import tempfile

from lt_engine.worker import serve
from lt_engine.workers._common import device


def load():
    from lt_engine.workers._common import limit_threads
    limit_threads()
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from lt_engine.pins import local                       # the pinned checkpoint, loaded from the cache
    return ChatterboxMultilingualTTS.from_local(local("ResembleAI/chatterbox"), device().split(":")[0])


def ref(m, audio, sr, text):
    import soundfile as sf
    path = os.path.join(tempfile.gettempdir(), "lt-chatterbox-ref.wav")
    sf.write(path, audio, sr)
    m.prepare_conditionals(path, exaggeration=0.3)


def say(m, text, lang):
    wav = m.generate(text, language_id=lang, cfg_weight=0.5, temperature=0.8)
    return wav.squeeze(0).cpu().numpy(), m.sr


if __name__ == "__main__":
    serve(load, ref, say)
