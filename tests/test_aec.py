"""Echo cancellation on a simulated room (skipped where the optional WebRTC package is not installed)."""
import numpy as np
import pytest

pytest.importorskip("livekit.rtc")


def test_the_translation_is_cancelled_and_the_person_kept():
    from lt_engine.aec import EchoCanceller
    sr, out_sr, n = 16000, 24000, 16000 * 12
    rng = np.random.default_rng(1)
    t = np.arange(n) / sr
    # "translation": speech-like noise bursts (we play it at 24 kHz); the person: a gliding tone from 8 s
    far16 = (rng.standard_normal(n) * (np.sin(2 * np.pi * 3 * t) > 0) * 0.2).astype(np.float32)
    person = np.where(t >= 8, 0.2 * np.sin(2 * np.pi * (300 + 40 * np.sin(2 * np.pi * t)) * t), 0).astype(np.float32)
    echo = np.zeros(n, np.float32)
    echo[64:] = 0.5 * far16[:-64]                             # 4 ms from speaker to microphone, half as loud
    mic = echo + person
    a = EchoCanceller(out_sr, delay_ms=10)
    cleaned = []
    far24 = np.interp(np.arange(0, n, sr / out_sr), np.arange(n), far16).astype(np.float32)
    for i in range(0, n, 512):
        a.played(far24[int(i * 1.5):int((i + 512) * 1.5)])    # the speakers play, then the microphone hears
        cleaned += a.process(mic[i:i + 512])
    out = np.concatenate(cleaned)

    def db(x):
        return 10 * np.log10(np.mean(x * x) + 1e-12)
    echo_only = slice(3 * sr, 7 * sr)
    assert db(mic[echo_only]) - db(out[echo_only]) > 15         # the echo is gone
    talk = slice(9 * sr, 11 * sr)
    assert db(out[talk]) > db(person[talk]) - 10                # the person is still there
