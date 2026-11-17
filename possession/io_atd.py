from __future__ import annotations

import numpy as np

import games

from .match import Match


def load_atd(game: str = games.GAME, data_dir: str | None = None) -> Match:
    from kloppy import metrica

    meta, raw = games.require(["atd_meta", "atd_raw"], game=game, data_dir=data_dir)
    ds = metrica.load_tracking_epts(meta_data=meta, raw_data=raw)
    df = ds.to_df()
    L, W = games.GAMES[game]["pitch"]

    rows = df["frame_id"].to_numpy(int) - 1
    T = max(games.source_frames(game=game), rows.max() + 1)

    def place(col):
        out = np.full(T, np.nan)
        out[rows] = df[col].to_numpy(float)
        return out

    ball = np.stack([(place("ball_x") - 0.5) * L, (place("ball_y") - 0.5) * W], axis=1)

    slot_team = {}
    for team in ds.metadata.teams:
        ground = str(getattr(team, "ground", "")).lower()
        t = 0 if "home" in ground else 1 if "away" in ground else -1
        for p in team.players:
            slot_team[str(p.player_id)] = t

    ids = [c[:-2] for c in df.columns if c.endswith("_x") and c != "ball_x"]
    players = np.full((T, len(ids), 2), np.nan)
    for k, pid in enumerate(ids):
        players[:, k, 0] = (place(f"{pid}_x") - 0.5) * L
        players[:, k, 1] = (place(f"{pid}_y") - 0.5) * W
    team = np.array([slot_team.get(pid, -1) for pid in ids])

    period = np.zeros(T, dtype=int)
    for p, a, b in games.half_video_blocks(game):
        period[a:b + 1] = p

    return Match(ball=ball, players=players, team=team, period=period, speed=None,
                 pitch_length=L, pitch_width=W, name=game)
