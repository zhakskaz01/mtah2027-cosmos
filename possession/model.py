from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

H, A, D, X = 0, 1, 2, 3


def swap_labels(y: np.ndarray) -> np.ndarray:
    return np.where(y == H, A, np.where(y == A, H, y))


class FrameModel:
    def __init__(self, **kw):
        params = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                      min_samples_leaf=200, l2_regularization=1.0, random_state=0)
        params.update(kw)
        self.clf = HistGradientBoostingClassifier(**params)
        self.columns = None
        self.prior = None

    def fit(self, frames: list[tuple[pd.DataFrame, np.ndarray]], stride: int = 5):
        Xs, ys = [], []
        for X_df, y in frames:
            keep = np.flatnonzero(y != X)[::stride]
            Xs.append(X_df.iloc[keep])
            ys.append(y[keep])
        Xc = pd.concat(Xs, ignore_index=True)
        yc = np.concatenate(ys)
        self.columns = list(Xc.columns)
        self.clf.fit(Xc, yc)
        self.prior = np.bincount(yc, minlength=3)[:3] / len(yc)
        return self

    def predict_proba(self, X_df: pd.DataFrame) -> np.ndarray:
        p = self.clf.predict_proba(X_df[self.columns])
        out = np.zeros((len(X_df), 3))
        out[:, self.clf.classes_] = p
        return out

    def predict_symmetric(self, feats: pd.DataFrame, feats_swapped: pd.DataFrame) -> np.ndarray:
        p = self.predict_proba(feats)
        q = self.predict_proba(feats_swapped)[:, [A, H, D]]
        return (p + q) / 2

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path) -> "FrameModel":
        with open(path, "rb") as fh:
            return pickle.load(fh)
