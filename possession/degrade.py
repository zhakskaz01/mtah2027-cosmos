from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .match import Match, FPS

D = 2


@dataclass
class DegradeConfig:
    live_ball_missing: tuple = (0.20, 0.40)
    gap_median_s: tuple = (6.0, 14.0)
    gap_sigma: float = 0.7
    gap_clip_s: tuple = (3.0, 90.0)
    snap_s: float = 1.0
    dead_full_p: tuple = (0.4, 0.9)
    dead_visible_p: tuple = (0.3, 0.8)
    dead_visible_s: tuple = (1.0, 6.0)
    ball_jitter_m: tuple = (0.1, 0.5)
    player_jitter_m: tuple = (0.05, 0.25)
    player_missing: tuple = (0.03, 0.10)
    false_ball_p: float = 0.004


def _long_gaps(T, live, rate, median_len, sigma, clip, snap, hard, rng):
    mask = np.zeros(T, bool)
    mean_gap = median_len * np.exp(sigma ** 2 / 2)
    mean_vis = mean_gap * (1 - rate) / max(rate, 1e-6)
    hard_idx = np.flatnonzero(hard & live)
    t = int(rng.exponential(mean_vis))
    while t < T:
        k = np.searchsorted(hard_idx, t)
        if k < len(hard_idx) and hard_idx[k] - t <= snap:
            t = int(hard_idx[k])
        g = int(np.clip(rng.lognormal(np.log(median_len), sigma), *clip))
        mask[t:t + g] = True
        t += g + int(rng.exponential(mean_vis))
    return mask & live


def _runs_mask(T, rate, mean_len, hazard, rng):
    mean_len = max(mean_len, 1.0)
    p_start = rate / mean_len * hazard / max(hazard.mean(), 1e-9)
    starts = np.flatnonzero(rng.random(T) < np.clip(p_start, 0, 1))
    lens = rng.geometric(1 / mean_len, size=len(starts))
    mask = np.zeros(T, bool)
    for s, l in zip(starts, lens):
        mask[s:s + l] = True
    return mask


def degrade(match: Match, labels: np.ndarray, seed: int, cfg: DegradeConfig = DegradeConfig()) -> Match:
    rng = np.random.default_rng(seed)
    u = lambda lohi: rng.uniform(*lohi)
    T = match.n_frames
    ball = match.ball.copy()
    players = match.players.copy()
    live = labels != D
    had_ball = ~np.isnan(ball[:, 0])


    restart_pos = np.full((T, 2), np.nan)
    nxt = None
    for t in range(T - 1, -1, -1):
        if had_ball[t] and live[t]:
            nxt = ball[t].copy()
        if not live[t] and nxt is not None:
            restart_pos[t] = nxt


    v = np.zeros(T)
    v[1:] = np.linalg.norm(np.diff(ball, axis=0), axis=1) * FPS
    v = np.nan_to_num(v)
    dist = np.linalg.norm(players - ball[:, None, :], axis=-1)
    crowd = np.nansum(dist < 3.0, axis=1)
    hard = (v > 12) | (crowd >= 3)

    already = np.isnan(ball[live, 0]).mean() if live.any() else 0.0
    rate = max(u(cfg.live_ball_missing) - already, 0.01)
    clip = (cfg.gap_clip_s[0] * FPS, cfg.gap_clip_s[1] * FPS)
    drop = _long_gaps(T, live, rate, u(cfg.gap_median_s) * FPS, cfg.gap_sigma, clip,
                      cfg.snap_s * FPS, hard, rng)
    ball[drop] = np.nan


    exit_pos = np.full((T, 2), np.nan)
    prv = None
    for t in range(T):
        if had_ball[t] and live[t]:
            prv = match.ball[t].copy()
        if not live[t] and prv is not None:
            exit_pos[t] = prv


    full_p = u(cfg.dead_full_p)
    part_p = u(cfg.dead_visible_p)
    L2, W2 = match.pitch_length / 2, match.pitch_width / 2
    ch = np.flatnonzero(np.diff(live.astype(int))) + 1
    starts, ends = np.r_[0, ch], np.r_[ch, T]
    for s, e in zip(starts, ends):
        if live[s] or e - s < 2:
            continue
        rest = restart_pos[s]
        if np.isnan(rest).any():
            continue
        r = rng.random()
        if r < full_p and not np.isnan(exit_pos[s]).any():
            ex = exit_pos[s].copy()

            if abs(ex[0]) > L2 - 3:
                ex[0] = np.sign(ex[0]) * (L2 + u((1.0, 4.0)))
            elif abs(ex[1]) > W2 - 3:
                ex[1] = np.sign(ex[1]) * (W2 + u((1.0, 4.0)))
            n = e - s
            hold = min(int(u(cfg.dead_visible_s) * FPS), n)
            move = n - hold
            w = (np.arange(move) / max(move, 1))[:, None]
            path = np.vstack([ex + w * (rest - ex), np.repeat(rest[None], hold, 0)])
            ball[s:e] = path + rng.normal(0, 0.3, path.shape)
            if rng.random() < 0.3 and move > 3 * FPS:
                g0 = s + int(rng.uniform(0.2, 0.6) * move)
                ball[g0:g0 + int(u((1.0, 3.0)) * FPS)] = np.nan
        elif r < full_p + (1 - full_p) * part_p:
            k = int(u(cfg.dead_visible_s) * FPS)
            a = max(s, e - k)
            ball[a:e] = rest + rng.normal(0, 0.3, (e - a, 2))


    ball += rng.normal(0, u(cfg.ball_jitter_m), ball.shape)
    players += rng.normal(0, u(cfg.player_jitter_m), players.shape)


    pm = u(cfg.player_missing)
    for k in range(players.shape[1]):
        m = _runs_mask(T, pm, 2.0 * FPS, np.ones(T), rng)
        players[m, k] = np.nan


    fb = (rng.random(T) < cfg.false_ball_p) & ~np.isnan(ball[:, 0])
    ball[fb] += rng.normal(0, 6.0, (fb.sum(), 2))

    return replace(match, ball=ball, players=players, speed=None,
                   name=f"{match.name}_deg{seed}")
