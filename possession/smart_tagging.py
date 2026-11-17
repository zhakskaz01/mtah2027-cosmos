from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from .match import FPS

ATTACKING = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION",
             "LONG BALL", "BALL IN FINAL THIRD", "SET PIECES"}
DEFENDING = {"DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN MIDDLE THIRD",
             "DEFENDING IN ATTACKING THIRD", "DEFENSIVE TRANSITION"}
HOME_NAMES = {"barcelona"}
AWAY_NAMES = {"atletic club", "athletic club"}


def load_instances(path) -> list[dict]:
    root = ET.parse(path).getroot()
    out = []
    for inst in root.iter("instance"):
        start = inst.findtext("start")
        end = inst.findtext("end")
        code = (inst.findtext("code") or "").strip()
        if start is None or end is None:
            continue
        tags = {}
        for lab in inst.findall("label"):
            g = (lab.findtext("group") or "").strip().lower()
            v = (lab.findtext("text") or "").strip()
            if g:
                tags[g] = v
        out.append({
            "start": int(round(float(start) * FPS)),
            "end": int(round(float(end) * FPS)),
            "code": code.upper(),
            "tags": tags,
        })
    return out


def team_index(name: str) -> int | None:
    n = name.strip().lower()
    if n in HOME_NAMES:
        return 0
    if n in AWAY_NAMES:
        return 1
    return None


def in_possession_team(inst: dict) -> int | None:
    t = team_index(inst["tags"].get("team", ""))
    if t is None:
        return None
    if inst["code"] in ATTACKING:
        return t
    if inst["code"] in DEFENDING:
        return 1 - t
    return None


def phase_evidence(instances, n_frames: int, weight: float) -> np.ndarray:
    ev = np.zeros((n_frames, 3))
    if weight <= 0:
        return ev


    says = np.zeros((n_frames, 2), dtype=bool)
    for inst in instances:
        t = in_possession_team(inst)
        if t is None:
            continue
        a, b = max(inst["start"], 0), min(inst["end"], n_frames - 1)
        says[a:b + 1, t] = True
    only_h = says[:, 0] & ~says[:, 1]
    only_a = says[:, 1] & ~says[:, 0]
    ev[only_h, 1] -= weight
    ev[only_a, 0] -= weight
    return ev
