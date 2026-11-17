from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FPS = 25
PITCH_LENGTH = 105.3
PITCH_WIDTH = 68.0


@dataclass
class Match:
    ball: np.ndarray

    players: np.ndarray

    team: np.ndarray

    period: np.ndarray

    speed: np.ndarray | None = None
    pitch_length: float = PITCH_LENGTH
    pitch_width: float = PITCH_WIDTH
    name: str = "match"
    extra: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return self.ball.shape[0]

    def half_blocks(self):
        out = []
        for p in (1, 2):
            idx = np.flatnonzero(self.period == p)
            if len(idx):
                out.append((p, int(idx[0]), int(idx[-1])))
        return out

    def player_speeds(self) -> np.ndarray:
        if self.speed is not None:
            return self.speed
        p = self.players
        d = np.full(p.shape[:2], np.nan)
        d[2:-2] = np.linalg.norm(p[4:] - p[:-4], axis=-1) / (4 / FPS)
        return d
