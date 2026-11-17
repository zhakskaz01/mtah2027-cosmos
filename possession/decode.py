from __future__ import annotations

from dataclasses import dataclass

import numpy as np

H, A, D, X = 0, 1, 2, 3


@dataclass
class DecoderParams:
    weight: float = 1.0
    switch: float = 12.0
    stop: float = 10.0
    restart: float = 10.0
    min_poss: int = 25


def _viterbi_py(logp: np.ndarray, trans: np.ndarray) -> np.ndarray:
    T, S = logp.shape
    back = np.zeros((T, S), dtype=np.int8)
    score = logp[0].copy()
    for t in range(1, T):
        cand = score[:, None] + trans
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(S)] + logp[t]
    path = np.empty(T, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _viterbi_loops(logp, trans):
    T, S = logp.shape
    back = np.zeros((T, S), dtype=np.int8)
    score = logp[0].copy()
    new = np.empty(S)
    for t in range(1, T):
        for j in range(S):
            best, arg = -1e300, 0
            for i in range(S):
                v = score[i] + trans[i, j]
                if v > best:
                    best, arg = v, i
            back[t, j] = arg
            new[j] = best + logp[t, j]
        score[:] = new
    path = np.empty(T, dtype=np.int8)
    path[-1] = np.argmax(score)
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


try:
    import numba
    viterbi = numba.njit(cache=True)(_viterbi_loops)
except ImportError:
    viterbi = _viterbi_py


def clean_short_runs(lab: np.ndarray, min_len: int) -> np.ndarray:
    lab = lab.copy()
    if min_len <= 0:
        return lab
    for _ in range(3):
        ch = np.flatnonzero(np.diff(lab)) + 1
        st, en = np.r_[0, ch], np.r_[ch, len(lab)]
        changed = False
        for i in range(1, len(st) - 1):
            a, b, c = lab[st[i - 1]], lab[st[i]], lab[st[i + 1]]
            if b in (H, A) and a == c and a in (H, A) and a != b and en[i] - st[i] < min_len:
                lab[st[i]:en[i]] = a
                changed = True
        if not changed:
            break
    return lab


def decode(proba: np.ndarray, blocks, prior: np.ndarray, p: DecoderParams,
           extra_logp: np.ndarray | None = None) -> np.ndarray:
    T = len(proba)
    out = np.full(T, X, dtype=np.int8)
    logp = p.weight * (np.log(np.clip(proba, 1e-6, 1)) - np.log(np.clip(prior, 1e-6, 1)))
    if extra_logp is not None:
        logp = logp + extra_logp
    trans = np.array([
        [0.0, -p.switch, -p.stop],
        [-p.switch, 0.0, -p.stop],
        [-p.restart, -p.restart, 0.0],
    ])
    for _, a, b in blocks:
        path = viterbi(logp[a:b + 1], trans)
        out[a:b + 1] = clean_short_runs(path, p.min_poss)
    return out
