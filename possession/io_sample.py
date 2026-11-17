from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .match import Match

SAMPLE_PITCH = (105.0, 68.0)


def _read_team(path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path, skiprows=2)

    cols = list(df.columns)
    names = []
    for i, c in enumerate(cols):
        if c.startswith("Player") or c == "Ball":
            names.append((c, cols[i], cols[i + 1]))
    return df, names


def load_sample_game(root: str | Path, game: int) -> Match:
    root = Path(root) / f"Sample_Game_{game}"
    home, hn = _read_team(root / f"Sample_Game_{game}_RawTrackingData_Home_Team.csv")
    away, an = _read_team(root / f"Sample_Game_{game}_RawTrackingData_Away_Team.csv")
    assert len(home) == len(away)
    L, W = SAMPLE_PITCH

    def xy(df, cx, cy):
        return np.stack([(df[cx].to_numpy(float) - 0.5) * L,
                         (df[cy].to_numpy(float) - 0.5) * W], axis=-1)

    slots, team = [], []
    for df, names, t in ((home, hn, 0), (away, an, 1)):
        for name, cx, cy in names:
            if name == "Ball":
                continue
            slots.append(xy(df, cx, cy))
            team.append(t)
    ball_name = [n for n in hn if n[0] == "Ball"][0]
    ball = xy(home, ball_name[1], ball_name[2])


    period = home["Period"].to_numpy(int)
    return Match(
        ball=ball,
        players=np.stack(slots, axis=1),
        team=np.array(team),
        period=period,
        pitch_length=L,
        pitch_width=W,
        name=f"sample{game}",
    )
