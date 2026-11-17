from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import games
from possession.decode import DecoderParams, decode
from possession.features import build_features
from possession.io_atd import load_atd
from possession.model import FrameModel
from possession.smart_tagging import load_instances, phase_evidence

LABELS = np.array(["H", "A", "D", "X"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--models", default="models")
    ap.add_argument("--out-dir", default="submission")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--tagging-weight", type=float, default=0.0)
    args = ap.parse_args()

    m = load_atd(data_dir=args.data_dir)
    print(f"loaded {m.n_frames} frames, {m.players.shape[1]} slots, "
          f"ball on {np.mean(~np.isnan(m.ball[:, 0])):.1%} of frames")

    model = FrameModel.load(Path(args.models) / "frame_model.pkl")
    params = DecoderParams(**json.loads((Path(args.models) / "decoder_params.json").read_text()))

    proba = model.predict_symmetric(build_features(m), build_features(m, swap=True))
    extra = None
    if args.tagging_weight > 0:
        (tag,) = games.require(["smart_tagging"], data_dir=args.data_dir)
        extra = phase_evidence(load_instances(tag), m.n_frames, args.tagging_weight)

    lab = LABELS[decode(proba, m.half_blocks(), model.prior, params, extra_logp=extra)]

    out = Path(args.out_dir) / f"{args.team}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines(f"{i},{l}\n" for i, l in enumerate(lab))
    counts = {k: int(np.sum(lab == k)) for k in "HADX"}
    print(f"wrote {out} ({len(lab)} rows) {counts}")


if __name__ == "__main__":
    main()
