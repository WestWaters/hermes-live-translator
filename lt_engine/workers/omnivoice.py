"""OmniVoice (CC-BY-NC-4.0 weights): 600+ languages. Default: a natural voice matched to the speaker's gender
(cross-language cloning carries an accent and is unreliable); pass --clone to clone. Env: transformers >= 5.3."""
import sys

import numpy as np

from lt_engine.worker import serve
from lt_engine.workers._common import device

REPO = next((a for a in sys.argv[1:] if not a.startswith("--")), "k2-fsa/OmniVoice")
CLONE = "--clone" in sys.argv
STEPS = 16


def load():
    from lt_engine.workers._common import limit_threads
    limit_threads()
    import torch
    from omnivoice import OmniVoice
    dev = device()
    from lt_engine.pins import local
    m = OmniVoice.from_pretrained(local(REPO), device_map=dev, dtype=torch.float16 if dev.startswith("cuda") else torch.float32)
    m.lt_prompt, m.lt_gender = None, None
    return m


def ref(m, audio, sr, text):
    from lt_engine.voices import speaker_gender
    a16 = audio if sr == 16000 else np.interp(np.linspace(0, len(audio) - 1, int(len(audio) * 16000 / sr)),
                                             np.arange(len(audio)), audio)
    m.lt_gender = speaker_gender(a16)
    if CLONE:
        import torch
        m.lt_prompt = m.create_voice_clone_prompt(ref_audio=(torch.from_numpy(np.asarray(audio, np.float32)), sr),
                                                  ref_text=text)


def say(m, text, lang):
    kw = {"voice_clone_prompt": m.lt_prompt} if m.lt_prompt is not None else (
        {"instruct": m.lt_gender} if m.lt_gender else {})
    audio = m.generate(text=text, language=lang, num_step=STEPS, **kw)[0]
    return np.asarray(audio, np.float32), m.sampling_rate


if __name__ == "__main__":
    serve(load, ref, say)
