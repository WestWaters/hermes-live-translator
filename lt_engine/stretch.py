"""Pitch-preserving speed-up (WSOLA) so translated speech can catch up with a fast speaker, like an interpreter
talking a little quicker. NumPy only."""
from __future__ import annotations

import numpy as np


def speed_up(x: np.ndarray, rate: float, sr: int = 24000) -> np.ndarray:
    """Return `x` played `rate` times faster (rate > 1) at the same pitch."""
    x = np.asarray(x, np.float32)
    if rate <= 1.01 or len(x) < sr // 10:
        return x
    n = int(sr * 0.03)                       # 30 ms frames
    hop_out = n // 2
    hop_in = int(round(hop_out * rate))
    tol = n // 4                             # search window for the best-aligned frame
    win = np.hanning(n).astype(np.float32)
    out_len = int(len(x) / rate) + n
    y = np.zeros(out_len + n, np.float32)
    w = np.zeros(out_len + n, np.float32)
    pos_in, pos_out = 0, 0
    prev = x[:n].copy()
    while pos_in + n + tol < len(x) and pos_out + n < len(y):
        if pos_out == 0:
            best = 0
        else:
            target = prev[hop_out:hop_out + n - hop_out]          # what naturally follows the last frame
            lo, hi = max(0, pos_in - tol), min(len(x) - n, pos_in + tol)
            best_c, best = -np.inf, pos_in
            for k in range(lo, hi, 4):
                c = float(np.dot(x[k:k + len(target)], target))
                if c > best_c:
                    best_c, best = c, k
        frame = x[best:best + n]
        y[pos_out:pos_out + n] += frame * win
        w[pos_out:pos_out + n] += win
        prev = x[best:best + n + hop_out] if best + n + hop_out <= len(x) else np.pad(frame, (0, hop_out))
        pos_in += hop_in
        pos_out += hop_out
    w[w < 1e-3] = 1.0
    return (y / w)[:pos_out + n // 2].astype(np.float32)
