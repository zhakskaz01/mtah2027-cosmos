from __future__ import annotations

import numpy as np

from .match import Match

H, A, D, X = 0, 1, 2, 3


def baseline(m: Match, half_width: int = 50) -> np.ndarray:
    d = np.linalg.norm(m.players - m.ball[:, None, :], axis=-1)
    d = np.where((m.team >= 0)[None, :], d, np.nan)
    has = ~np.all(np.isnan(d), axis=1)
    near = np.argmin(np.where(np.isnan(d), np.inf, d), axis=1)
    code = np.where(has, m.team[near], -1)
    if half_width <= 0:
        lab = np.where(code == 0, H, np.where(code == 1, A, D))
    else:
        T = len(code)
        ch = np.r_[0, np.cumsum(code == 0)]
        ca = np.r_[0, np.cumsum(code == 1)]
        i = np.arange(T)
        lo, hi = np.maximum(0, i - half_width), np.minimum(T, i + half_width + 1)
        h, a = ch[hi] - ch[lo], ca[hi] - ca[lo]
        lab = np.where(h + a == 0, D, np.where(h >= a, H, A))
    out = np.full(m.n_frames, X, dtype=np.int8)
    for _, s, e in m.half_blocks():
        out[s:e + 1] = lab[s:e + 1]
    return out
