from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from possession.baseline import baseline
from possession.decode import DecoderParams, decode
from possession.degrade import DegradeConfig, degrade
from possession.features import build_features
from possession.io_sample import load_sample_game
from possession.labels_events import labels_from_events, load_events
from possession.metrics import summary, spell_f1
from possession.model import FrameModel, swap_labels

GAMES = (1, 2)
GRID = {
    "switch": [2, 4, 8, 12, 18, 26, 40],
    "stop": [4, 8, 12, 18, 26, 40, 60],
    "restart": [4, 8, 12, 18, 26, 40, 60],
    "weight": [0.5, 0.75, 1.0, 1.5],
    "min_poss": [0, 12, 25, 50],
}
STRESS = DegradeConfig(live_ball_missing=(0.45, 0.5), gap_median_s=(16.0, 22.0),
                       dead_full_p=(0.95, 1.0), dead_visible_p=(0.9, 0.95), dead_visible_s=(4.0, 10.0),
                       player_missing=(0.12, 0.15), false_ball_p=0.01)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def prepare(root, copies):
    data = {}
    for g in GAMES:
        m = load_sample_game(root, g)
        y = labels_from_events(m, load_events(root, g))
        reps = []
        for c in range(copies):
            md = degrade(m, y, seed=1000 * g + c)
            reps.append((md, build_features(md), build_features(md, swap=True)))
        data[g] = (m, y, reps)
        log(f"sample game {g}: {copies} degraded copies")
    return data


def fit(data, games):
    rows = []
    for g in games:
        _, y, reps = data[g]
        for _, F, Fs in reps:
            rows += [(F, y), (Fs, swap_labels(y))]
    return FrameModel().fit(rows)


def score(probs, data, games, params, prior):
    s = []
    for g in games:
        m, y, _ = data[g]
        for p in probs[g]:
            s.append(spell_f1(decode(p, m.half_blocks(), prior, params), y, m.half_blocks()))
    return float(np.mean(s))


def tune(probs, data, games, prior, rounds=2):
    best = DecoderParams()
    best_s = score(probs, data, games, best, prior)
    for _ in range(rounds):
        for k, vals in GRID.items():
            for v in vals:
                cand = replace(best, **{k: v})
                s = score(probs, data, games, cand, prior)
                if s > best_s + 1e-4:
                    best, best_s = cand, s
        log(f"tuned on games {list(games)}: spellF1 {best_s:.4f} {asdict(best)}")
    return best


def avg(rs):
    return {k: round(float(np.mean([r[k] for r in rs])), 4) for k in rs[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-data", default="external/sample-data/data")
    ap.add_argument("--copies", type=int, default=3)
    ap.add_argument("--out", default="models")
    args = ap.parse_args()

    data = prepare(args.sample_data, args.copies)

    probs, priors, models = {}, {}, {}
    for test in GAMES:
        trn = [g for g in GAMES if g != test]
        model = fit(data, trn)
        priors[test], models[test] = model.prior, model
        probs[test] = [model.predict_symmetric(F, Fs) for _, F, Fs in data[test][2]]
        log(f"trained on game {trn}, predicted game {test}")

    report = {"held_out": {}}
    for test in GAMES:
        params = tune(probs, data, [g for g in GAMES if g != test], priors[test])
        m, y, reps = data[test]
        hb = m.half_blocks()
        ours = [summary(decode(p, hb, priors[test], params), y, hb) for p in probs[test]]
        argmax = [summary(np.where(m.period > 0, p.argmax(1), 3).astype(np.int8), y, hb) for p in probs[test]]
        base = {w: [summary(baseline(md, w), y, hb) for md, _, _ in reps] for w in (0, 25, 50, 125)}
        stress = []
        for c in range(2):
            md = degrade(m, y, seed=50_000 + 10 * test + c, cfg=STRESS)
            p = models[test].predict_symmetric(build_features(md), build_features(md, swap=True))
            stress.append((summary(decode(p, hb, priors[test], params), y, hb), summary(baseline(md, 50), y, hb)))
        report["held_out"][f"game{test}"] = {
            "stress_ours": avg([a for a, _ in stress]),
            "stress_baseline_w50": avg([b for _, b in stress]),
            "decoder_params_tuned_on_other_game": asdict(params),
            "ours": avg(ours),
            "classifier_argmax_no_decoder": avg(argmax),
            **{f"baseline_w{w}": avg(r) for w, r in base.items()},
        }
        log(f"game {test} held out:", json.dumps(report["held_out"][f"game{test}"]["ours"]))

    final = fit(data, GAMES)
    params = tune(probs, data, GAMES, np.mean([priors[g] for g in GAMES], axis=0))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    final.save(out / "frame_model.pkl")
    report["final_decoder_params"] = asdict(params)
    (out / "decoder_params.json").write_text(json.dumps(asdict(params), indent=2))
    (out / "sample_report.json").write_text(json.dumps(report, indent=2))
    log("saved", *(str(out / f) for f in ("frame_model.pkl", "decoder_params.json", "sample_report.json")))


if __name__ == "__main__":
    main()
