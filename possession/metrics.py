from __future__ import annotations

import numpy as np

H, A, D, X = 0, 1, 2, 3
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)


def macro_f1(pred: np.ndarray, truth: np.ndarray) -> float:
    g = truth != X
    p, t = pred[g], truth[g]
    f1s = []
    for c in (H, A, D):
        tp = np.sum((p == c) & (t == c))
        fp = np.sum((p == c) & (t != c))
        fn = np.sum((p != c) & (t == c))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def fill_abstentions(pred: np.ndarray, blocks) -> np.ndarray:
    out = pred.copy()
    for _, a, b in blocks:
        seg = out[a:b + 1]
        idx = np.where(seg != X, np.arange(len(seg)), -1)
        idx = np.maximum.accumulate(idx)
        first = np.flatnonzero(seg != X)
        if len(first) == 0:
            continue
        idx[idx < 0] = first[0]
        out[a:b + 1] = seg[idx]
    return out


def segments(lab: np.ndarray, offset: int = 0):
    out = []
    if len(lab) == 0:
        return out
    change = np.flatnonzero(np.diff(lab)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(lab)]
    for s, e in zip(starts, ends):
        if lab[s] in (H, A):
            out.append((int(lab[s]), s + offset, e + offset))
    return out


def _f1_at(pred_segs, true_segs, thr):
    pairs = []
    j = 0

    for pi, (pl, ps, pe) in enumerate(pred_segs):
        while j < len(true_segs) and true_segs[j][2] <= ps:
            j += 1
        k = j
        while k < len(true_segs) and true_segs[k][1] < pe:
            tl, ts, te = true_segs[k]
            if tl == pl:
                inter = min(pe, te) - max(ps, ts)
                union = max(pe, te) - min(ps, ts)
                iou = inter / union
                if iou >= thr:
                    pairs.append((iou, pi, k))
            k += 1
    pairs.sort(reverse=True)
    used_p, used_t, tp = set(), set(), 0
    for iou, pi, ti in pairs:
        if pi not in used_p and ti not in used_t:
            used_p.add(pi)
            used_t.add(ti)
            tp += 1
    n = len(pred_segs) + len(true_segs)
    return 0.0 if n == 0 else 2 * tp / n


def spell_f1(pred: np.ndarray, truth: np.ndarray, blocks) -> float:
    filled = fill_abstentions(pred, blocks)
    ps, ts = [], []
    for _, a, b in blocks:
        g = truth[a:b + 1] != X
        if not g.any():
            continue

        idx = np.flatnonzero(g)
        lo, hi = a + idx[0], a + idx[-1] + 1
        ps += segments(filled[lo:hi], lo)
        ts += segments(truth[lo:hi], lo)
    return float(np.mean([_f1_at(ps, ts, t) for t in THRESHOLDS]))


def summary(pred, truth, blocks) -> dict:
    filled = fill_abstentions(pred, blocks)
    g = truth != X
    n_pred = sum(len(segments(filled[a:b + 1])) for _, a, b in blocks)
    n_true = sum(len(segments(np.where(g[a:b + 1], truth[a:b + 1], D))) for _, a, b in blocks)
    return {
        "spellF1": round(spell_f1(pred, truth, blocks), 4),
        "macroF1": round(macro_f1(pred, truth), 4),
        "n_poss_pred": n_pred,
        "n_poss_true": n_true,
    }
