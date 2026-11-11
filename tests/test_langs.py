import pytest

from lt_engine import langs


def test_names_and_codes_resolve():
    assert langs.resolve("Spanish") == "es" and langs.resolve("zh") == "zh"
    with pytest.raises(ValueError):
        langs.resolve("Klingon")


def test_fast_tables_are_subsets_of_what_is_understood():
    assert langs.FAST_ASR <= set(langs.NAMES) and langs.FAST_CLONE <= langs.CLONE_LANGS


class _Det:
    def __init__(self, w, wp, q):
        self.w, self.wp, self.q = w, wp, q


def _vote(w, wp, q, seconds=4.0):
    import numpy as np
    from lt_engine import asr
    r = asr.Recognizer.__new__(asr.Recognizer)
    r.whisper = type("W", (), {"detect": lambda self, a: (w, wp)})()
    r.qwen = type("Q", (), {"transcribe": lambda self, a, lang=None: ("", asr.QWEN_BY_CODE.get(q, q) and q)})()
    r.route, r.made, r.engine, r.language = [({"*"}, "whisper")], {"whisper": r.whisper}, None, None
    r._pick = lambda: None
    ok = r._identify(np.zeros(int(16000 * seconds), np.float32))
    return r.language if ok else "wait"


def test_background_voices_cannot_hijack_the_language():
    assert _vote("fr", 0.7, "zh") == "zh"          # Whisper hears French chatter, Qwen3-ASR hears the Chinese speaker
    assert _vote("zh", 0.95, "zh") == "zh"         # agreement
    assert _vote("sw", 0.95, "en") == "sw"         # Swahili: outside Qwen3-ASR's 30, Whisper is sure
    assert _vote("sw", 0.5, "en") == "wait"        # unsure: hear more first
    assert _vote("sw", 0.5, "en", seconds=8) == "en"
    assert _vote("ur", 0.96, "hi") == "hi"         # spoken Hindi and Urdu sound alike: Qwen3-ASR decides


def test_hallucinations_dropped_real_speech_kept():
    from lt_engine.pipeline import hallucinated
    assert hallucinated("Thank you for watching!")
    assert hallucinated("Subtítulos por la comunidad de Amara.org")
    assert hallucinated("you you you you")
    assert not hallucinated("Thank you, that was very helpful.")
    assert not hallucinated("The report warns that nobody can guarantee it.")


def test_runaway_voice_cap_is_generous_but_finite():
    from lt_engine.pipeline import max_speech_s
    es = "Nadie puede garantizar que las acciones tomadas ahora en Irak detengan la guerra."
    assert 9 < max_speech_s(es, "es") < 20                 # ~5 s to say; a 460 s runaway is cut at ~14 s
    assert 5 < max_speech_s("没有人能保证。", "zh") < 10


def _recheck(heard, p, seconds):
    import numpy as np
    from lt_engine import asr
    r = asr.Recognizer.__new__(asr.Recognizer)
    r.whisper = type("W", (), {"detect": lambda self, a: (heard, p)})()
    r.engine = type("E", (), {"transcribe": lambda self, a, lang=None: ("text", lang)})()
    r.fixed, r.language, r.held, r.committed, r.prev, r.qwen = None, "zh", [], "", None, None
    r._pick = lambda: None
    r.utterance()
    r._run(np.zeros(int(16000 * seconds), np.float32))
    return r.language


def test_language_switch_needs_a_confident_long_enough_phrase():
    assert _recheck("ja", 0.95, 3.0) == "ja"        # someone now speaks Japanese
    assert _recheck("ja", 0.6, 3.0) == "zh"         # not sure: keep the language
    assert _recheck("ja", 0.95, 1.0) == "zh"        # "はい": too short to switch on


def test_short_music_hallucination_dropped_but_words_containing_it_kept():
    from lt_engine.pipeline import hallucinated
    assert hallucinated("موسیقی")
    assert hallucinated("(Music)")
    assert not hallucinated("I love the music of this city, it is wonderful.")


def test_quiet_lone_words_are_background_but_quiet_sentences_and_loud_words_stay():
    from types import SimpleNamespace as NS
    from lt_engine.pipeline import stray
    ln = NS(level=-20.0, phrase_db=-34.0, rec=NS(language="en"))
    assert stray(ln, "Okay.")
    assert not stray(ln, "Okay, let us start the meeting now.")
    ln.phrase_db = -22.0
    assert not stray(ln, "Yes.")                   # the speaker's own short answer


def test_builtin_translator_follows_the_hardware(monkeypatch):
    import builtins
    from lt_engine import hw, mt
    real_import = builtins.__import__

    def no_llama(name, *a, **k):
        if name == "llama_cpp":
            raise ImportError(name)
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_llama)
    for backend, spec in (("mlx", mt.DEFAULT), ("cuda", mt.DEFAULT_TORCH), ("cpu", mt.DEFAULT_GGUF)):
        monkeypatch.setattr(hw, "detect", lambda b=backend: b)
        assert mt.default_spec() == spec               # ARM64 + NVIDIA without llama.cpp: transformers on the GPU
