from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from .match import Match, FPS

CONTROL_M = 2.0
MAX_INTERP = int(1.5 * FPS)
FAR = 30.0
SCALES = (12, 25, 50, 100)


def _interp_ball(ball: np.ndarray, max_gap: int):
    T = len(ball)
    ok = ~np.isnan(ball[:, 0])
    out = ball.copy()
    idx = np.flatnonzero(ok)
    if len(idx) < 2:
        return out, ok.copy()
    gaps = np.flatnonzero(np.diff(idx) > 1)
    for g in gaps:
        a, b = idx[g], idx[g + 1]
        if b - a - 1 <= max_gap:
            w = (np.arange(a + 1, b) - a) / (b - a)
            out[a + 1:b] = ball[a] + w[:, None] * (ball[b] - ball[a])
    return out, ~np.isnan(out[:, 0])


def _gap_length(present: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = len(present)
    since = np.zeros(T)
    c = 0
    for t in range(T):
        c = 0 if present[t] else c + 1
        since[t] = c
    run = np.zeros(T)
    ch = np.flatnonzero(np.diff(present.astype(int))) + 1
    for s, e in zip(np.r_[0, ch], np.r_[ch, T]):
        if not present[s]:
            run[s:e] = e - s
    return run, since


def _roll(x, w):
    v = np.nan_to_num(x)
    n = (~np.isnan(x)).astype(float)
    s = uniform_filter1d(v, w, mode="nearest")
    c = uniform_filter1d(n, w, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 1e-9, s / c, np.nan)


def _roll_side(x, w, side):
    v = np.nan_to_num(x)
    cs = np.r_[0, np.cumsum(v)]
    cn = np.r_[0, np.cumsum(~np.isnan(x))]
    T = len(x)
    t = np.arange(T)
    if side < 0:
        lo, hi = np.maximum(t - w, 0), t
    else:
        lo, hi = t + 1, np.minimum(t + 1 + w, T)
    s, c = cs[hi] - cs[lo], cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / c, np.nan)


def _smooth_positions(p, w=5):
    v = np.nan_to_num(p)
    n = (~np.isnan(p)).astype(float)
    s = uniform_filter1d(v, w, axis=0, mode="nearest")
    c = uniform_filter1d(n, w, axis=0, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(c > 0.5, s / np.maximum(c, 1e-9), np.nan)
    return np.where(np.isnan(p), np.nan, out)


def _speed(p, half=5):
    ps = _smooth_positions(p)
    sp = np.full(p.shape[:-1], np.nan)
    sp[half:-half] = np.linalg.norm(ps[2 * half:] - ps[:-2 * half], axis=-1) / (2 * half / FPS)
    return sp


def build_features(m: Match, swap: bool = False) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _build(m, swap)


def _build(m: Match, swap: bool) -> pd.DataFrame:
    T = m.n_frames
    team = m.team.copy()
    if swap:
        team = np.where(team == 0, 1, np.where(team == 1, 0, team))
    home, away = team == 0, team == 1
    L2, W2 = m.pitch_length / 2, m.pitch_width / 2
    f = {}


    present = ~np.isnan(m.ball[:, 0])
    bi, has_bi = _interp_ball(m.ball, MAX_INTERP)
    gap_run, since = _gap_length(present)
    f["ball_present"] = present.astype(float)
    f["ball_gap_s"] = np.minimum(gap_run, 40 * FPS) / FPS
    f["ball_since_s"] = np.minimum(since, 40 * FPS) / FPS
    bspeed = _speed(bi[:, None, :])[:, 0]
    f["ball_speed"] = np.clip(np.nan_to_num(bspeed, nan=-1.0), -1, 40)
    out = (np.abs(bi[:, 0]) > L2 + 0.5) | (np.abs(bi[:, 1]) > W2 + 0.5)
    out = out & has_bi
    f["ball_out"] = out.astype(float)

    last = pd.DataFrame(bi).ffill().to_numpy()
    f["ball_absx"] = np.nan_to_num(np.abs(last[:, 0]) / L2, nan=0.0)
    f["ball_absy"] = np.nan_to_num(np.abs(last[:, 1]) / W2, nan=0.0)
    static = ((bspeed < 0.8) & has_bi).astype(float)


    line_d = np.minimum(L2 - np.abs(bi[:, 0]), W2 - np.abs(bi[:, 1]))
    line_d = np.where(has_bi, np.clip(line_d, -10, 20), 20.0)
    f["ball_line_dist"] = line_d
    slow = has_bi & (np.nan_to_num(bspeed, nan=99) < 3.0)
    slow_line = (slow & (line_d < 2.5)).astype(float)
    moving = has_bi & (np.nan_to_num(bspeed, nan=0) > 4.0)
    mv = np.where(moving, np.arange(T), np.nan)
    since_moving = np.arange(T) - pd.Series(mv).ffill().fillna(-60 * FPS).to_numpy()
    until_moving = pd.Series(mv).bfill().fillna(T + 60 * FPS).to_numpy() - np.arange(T)
    f["ball_since_moving_s"] = np.minimum(since_moving, 60 * FPS) / FPS
    f["ball_until_moving_s"] = np.minimum(until_moving, 60 * FPS) / FPS
    for w in (50, 125, 250):
        f[f"ball_line_dist_min_{w}"] = (
            pd.Series(line_d).rolling(w, center=True, min_periods=1).min().to_numpy())
        f[f"ball_slow_line_{w}"] = _roll(slow_line, w)
        f[f"ball_slow_{w}"] = _roll(slow.astype(float), w)


    d = np.linalg.norm(m.players - bi[:, None, :], axis=-1)
    dh = np.nanmin(np.where(home[None, :], d, np.inf), axis=1)
    da = np.nanmin(np.where(away[None, :], d, np.inf), axis=1)
    dh = np.where(np.isfinite(dh), dh, np.nan)
    da = np.where(np.isfinite(da), da, np.nan)
    dh_f, da_f = np.nan_to_num(dh, nan=FAR), np.nan_to_num(da, nan=FAR)
    f["own_dh"] = np.minimum(dh_f, FAR)
    f["own_da"] = np.minimum(da_f, FAR)
    diff = np.clip(dh_f - da_f, -FAR, FAR)
    f["own_diff"] = diff
    hc = ((dh_f < CONTROL_M) & (dh_f < da_f)).astype(float)
    ac = ((da_f < CONTROL_M) & (da_f < dh_f)).astype(float)

    hc_n = np.where(has_bi, hc, np.nan)
    ac_n = np.where(has_bi, ac, np.nan)
    diff_n = np.where(has_bi, diff, np.nan)
    near_h = np.sum(np.where(home[None, :], d < 5, False), axis=1)
    near_a = np.sum(np.where(away[None, :], d < 5, False), axis=1)
    f["own_near5_diff"] = np.where(has_bi, near_h - near_a, 0)


    lc = np.where(hc > 0, 1.0, np.where(ac > 0, -1.0, np.nan))
    lc_ff = pd.Series(lc).ffill().fillna(0).to_numpy()
    lc_bf = pd.Series(lc).bfill().fillna(0).to_numpy()
    f["own_last_ctrl"] = lc_ff
    f["own_next_ctrl"] = lc_bf
    idx_last = pd.Series(np.where(~np.isnan(lc), np.arange(T), np.nan)).ffill().fillna(-10 * FPS).to_numpy()
    idx_next = pd.Series(np.where(~np.isnan(lc), np.arange(T), np.nan)).bfill().fillna(T + 10 * FPS).to_numpy()
    f["own_last_ctrl_age_s"] = np.minimum(np.arange(T) - idx_last, 20 * FPS) / FPS
    f["own_next_ctrl_in_s"] = np.minimum(idx_next - np.arange(T), 20 * FPS) / FPS

    for w in SCALES:
        f[f"own_hc_{w}"] = np.nan_to_num(_roll(hc_n, w), nan=0.0)
        f[f"own_ac_{w}"] = np.nan_to_num(_roll(ac_n, w), nan=0.0)
        f[f"own_diff_{w}"] = np.nan_to_num(_roll(diff_n, w), nan=0.0)
        f[f"ball_present_{w}"] = _roll(present.astype(float), w)
        for side, tag in ((-1, "past"), (1, "next")):
            hcs = _roll_side(hc_n, w, side)
            acs = _roll_side(ac_n, w, side)
            f[f"own_ctrl_{tag}_{w}"] = np.nan_to_num(hcs - acs, nan=0.0)


    sp = _speed(m.players)
    sp = np.where(sp > 12, np.nan, sp)
    with np.errstate(all="ignore"):
        mov_med = np.nanmedian(sp, axis=1)
        mov_slow = np.nanmean(np.where(np.isnan(sp), np.nan, sp < 1.5), axis=1)
        mov_h = np.nanmean(np.where(home[None, :], sp, np.nan), axis=1)
        mov_a = np.nanmean(np.where(away[None, :], sp, np.nan), axis=1)
    mov_med = np.nan_to_num(mov_med, nan=0.0)
    mov_slow = np.nan_to_num(mov_slow, nan=1.0)
    f["mov_n_players"] = np.sum(~np.isnan(m.players[:, :, 0]), axis=1).astype(float)
    for w in SCALES + (250,):
        f[f"mov_med_{w}"] = _roll(mov_med, w)
        f[f"mov_slow_{w}"] = _roll(mov_slow, w)
        f[f"ball_out_{w}"] = _roll(out.astype(float), w)
        f[f"ball_static_{w}"] = _roll(static, w)
    for w in (25, 75, 250):
        f[f"mov_accel_{w}"] = np.nan_to_num(_roll_side(mov_med, w, +1) - _roll_side(mov_med, w, -1))
        f[f"ball_out_past_{w}"] = np.nan_to_num(_roll_side(out.astype(float), w, -1))
        f[f"ball_present_past_{w}"] = np.nan_to_num(_roll_side(present.astype(float), w, -1), nan=0.0)
        f[f"ball_present_next_{w}"] = np.nan_to_num(_roll_side(present.astype(float), w, +1), nan=0.0)
    f["mov_team_diff_50"] = np.nan_to_num(_roll(mov_h - mov_a, 50))


    def spread(mask):
        p = np.where(mask[None, :, None], m.players, np.nan)
        with np.errstate(all="ignore"):
            return np.nanstd(p[:, :, 0], axis=1) + np.nanstd(p[:, :, 1], axis=1)
    sh, sa = spread(home), spread(away)
    for w in (50, 125):
        f[f"shape_spread_diff_{w}"] = np.nan_to_num(_roll(sh - sa, w))
        f[f"shape_spread_sum_{w}"] = np.nan_to_num(_roll(sh + sa, w))


    with np.errstate(all="ignore"):
        hx = np.nanmean(np.where(home[None, :], m.players[:, :, 0], np.nan), axis=1)
        ax = np.nanmean(np.where(away[None, :], m.players[:, :, 0], np.nan), axis=1)
        allx = np.nanmean(np.where((home | away)[None, :], m.players[:, :, 0], np.nan), axis=1)
    direction = np.zeros(T)
    for _, a, b in m.half_blocks():
        d_half = np.nanmedian(hx[a:b + 1] - ax[a:b + 1])
        direction[a:b + 1] = -np.sign(d_half) if np.isfinite(d_half) else 0.0

    terr = direction * allx / L2
    vel = np.zeros(T)
    vel[1:] = np.diff(allx) * FPS
    vel = direction * np.clip(np.nan_to_num(vel), -10, 10)


    u = direction[:, None] * m.players[:, :, 0]
    u_home = np.sort(np.where(home[None, :], u, np.nan), axis=1)[:, 1]
    u_away = -np.sort(np.where(away[None, :], -u, np.nan), axis=1)[:, 1]
    line = (u_home + L2) - (L2 - u_away)
    ball_terr = direction * last[:, 0] / L2
    f["dir_ball_terr"] = np.nan_to_num(ball_terr)
    for w in (25, 100, 250):
        f[f"dir_terr_{w}"] = np.nan_to_num(_roll(terr, w))
        f[f"dir_line_{w}"] = np.nan_to_num(_roll(line, w))
    for w in (25, 75, 200):
        f[f"dir_vel_{w}"] = np.nan_to_num(_roll(vel, w))
        f[f"dir_vel_past_{w}"] = np.nan_to_num(_roll_side(vel, w, -1))
        f[f"dir_vel_next_{w}"] = np.nan_to_num(_roll_side(vel, w, +1))

    df = pd.DataFrame(f)
    return df.fillna(0.0).astype(np.float32)
