import numpy as np

from lt_engine.stretch import speed_up


def test_speed_up_shortens_by_the_rate_and_keeps_the_signal():
    sr = 24000
    t = np.arange(sr * 3) / sr
    x = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    y = speed_up(x, 1.25, sr)
    assert abs(len(y) / len(x) - 1 / 1.25) < 0.03
    assert abs(np.sqrt((y ** 2).mean()) - np.sqrt((x ** 2).mean())) < 0.05     # same loudness


def test_no_speed_up_returns_the_input():
    x = np.ones(24000, np.float32)
    assert speed_up(x, 1.0) is x
