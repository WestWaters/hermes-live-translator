"""Stand-in for python-soxr on Windows on Arm, where it has no wheel and building it needs a C++ compiler.
librosa (pulled in by the voice models) only calls soxr.resample(); polyphase resampling from scipy does that job."""
from math import gcd

import numpy as np

__version__ = "1.1.0+live-translate-compat"


def resample(x, in_rate, out_rate, quality="HQ"):
    from scipy.signal import resample_poly
    x = np.asarray(x)
    g = gcd(int(in_rate), int(out_rate))
    return resample_poly(x, int(out_rate) // g, int(in_rate) // g, axis=0).astype(x.dtype, copy=False)
