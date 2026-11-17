from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .match import Match

OWN_TYPES = {"PASS", "SET PIECE", "RECOVERY", "SHOT", "BALL LOST", "BALL OUT"}
NOT_A_RESTART = {"CARD", "FAULT RECEIVED", "CHALLENGE"}

H, A, D, X = 0, 1, 2, 3
LABELS = np.array(["H", "A", "D", "X"])


def load_events(root: str | Path, game: int) -> pd.DataFrame:
    p = Path(root) / f"Sample_Game_{game}" / f"Sample_Game_{game}_RawEventsData.csv"
    ev = pd.read_csv(p)
    ev["Subtype"] = ev["Subtype"].fillna("")

    ev["s"] = ev["Start Frame"].astype(int) - 1
    ev["e"] = ev["End Frame"].astype(int) - 1
    ev.loc[ev["e"] < ev["s"], "e"] = ev["s"]
    return ev.reset_index(drop=True)


def _next_restart(ev: pd.DataFrame, i: int, after: int, set_piece_only: bool) -> int | None:
    later = ev.iloc[i + 1:]
    later = later[later["s"] > after]
    if set_piece_only:
        sp = later[later["Type"] == "SET PIECE"]
        if len(sp):
            return int(sp["s"].iloc[0])
    later = later[~later["Type"].isin(NOT_A_RESTART)]
    return int(later["s"].iloc[0]) if len(later) else None


def labels_from_events(match: Match, ev: pd.DataFrame, min_control: int = 25) -> np.ndarray:
    T = match.n_frames
    y = np.full(T, X, dtype=np.int8)
    for period, first, last in match.half_blocks():
        e = ev[ev["Period"] == period].sort_values("s", kind="stable").reset_index(drop=True)


        owner = np.full(last - first + 1, -1, dtype=np.int8)
        own = e[e["Type"].isin(OWN_TYPES)]
        for _, r in own.iterrows():
            owner[max(r["s"] - first, 0):] = H if r["Team"] == "Home" else A
        seg = owner.copy()


        dead = np.zeros_like(owner, dtype=bool)
        dead[: max(e["s"].iloc[0] - first, 0)] = True
        for i, r in e.iterrows():
            t, st = r["Type"], r["Subtype"]
            if t == "BALL OUT" or (t == "SHOT" and ("OUT" in st or "GOAL" in st)):
                start = r["e"]
                end = _next_restart(e, i, start, set_piece_only=False)
            elif t == "FAULT RECEIVED" or (t == "CHALLENGE" and "FAULT" in st and "ADVANTAGE" not in st):
                start = r["s"]
                end = _next_restart(e, i, start, set_piece_only=True)
            elif t == "BALL LOST" and "OFFSIDE" in st:
                start = r["e"]
                end = _next_restart(e, i, start, set_piece_only=True)
            elif t == "BALL LOST" and "END HALF" in st:
                start, end = r["s"], last + 1
            else:
                continue
            if end is None:
                end = last + 1
            dead[max(start - first, 0): max(end - first, 0)] = True
        seg[dead] = D
        seg[seg < 0] = D
        y[first:last + 1] = absorb_uncontrolled(seg, min_control)
    return y


def absorb_uncontrolled(seg: np.ndarray, min_control: int) -> np.ndarray:
    if min_control <= 0:
        return seg
    seg = seg.copy()
    changed = True
    while changed:
        changed = False
        ch = np.flatnonzero(np.diff(seg)) + 1
        st, en = np.r_[0, ch], np.r_[ch, len(seg)]
        for i in range(1, len(st) - 1):
            lab, prev, nxt = seg[st[i]], seg[st[i - 1]], seg[st[i + 1]]
            if lab in (H, A) and prev == nxt and prev in (H, A) and prev != lab \
                    and en[i] - st[i] < min_control:
                seg[st[i]:en[i]] = prev
                changed = True
                break
    return seg
