"""Closed-set drone-model classification on CardRF, with two kinds of held-out data.

    python scripts/run_closed_set.py --seeds 3

A. Time-block split: every recording is cut into train / validation / test blocks with
   gaps (see spectrumsentry.data). Tests unseen *moments* of recordings the model has seen.
B. Cross-recording split: for the four drones recorded in two files, train on part 1 (and
   on the train blocks of every other recording) and test on part 2. Tests an unseen
   *recording* of the same drone, the nearest this dataset gets to a different session.

Windows are 65,536 IQ samples (0.55 ms at 120 MS/s). Everything here is a single
anechoic chamber with one receiver; see docs/design.md.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from spectrumsentry.catalog import catalog
from spectrumsentry.data import Store, block_spans, class_names, eval_windows, whole_spans
from spectrumsentry.train import accuracy, predict, train

ROOT = Path(__file__).resolve().parents[1]


def per_class(logits, labels, names):
    pred = logits.argmax(1)
    return {n: round((pred[labels == i] == i).float().mean().item(), 3) for i, n in enumerate(names) if (labels == i).any()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--per-span", type=int, default=60)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    recs = catalog(ROOT / "data" / "cardrf")
    names = class_names(recs)
    store = Store()
    two_file = sorted({r.drone for r in recs if r.part})
    print(f"{len(recs)} recordings, {len(names)} drones; two-file drones: {two_file}", flush=True)

    out = {"classes": names, "two_file_drones": two_file, "steps": args.steps, "device": device, "runs": []}
    for seed in range(args.seeds):
        t0 = time.time()
        # A: time blocks over all recordings
        model = train(block_spans(recs, names, "train"), len(names), store, device, steps=args.steps, seed=seed)
        test_spans = block_spans(recs, names, "test")
        logits, labels = predict(model, eval_windows(test_spans, args.per_span), store, device)
        a = {
            "val": accuracy(model, block_spans(recs, names, "val"), store, device, args.per_span),
            "test": (logits.argmax(1) == labels).float().mean().item(),
            "per_class_test": per_class(logits, labels, names),
        }
        # B: cross-recording (train on part 0/1 only; test on part 2)
        modelb = train(block_spans(recs, names, "train", parts=(0, 1)), len(names), store, device, steps=args.steps, seed=seed)
        held = whole_spans(recs, names, parts=(2,))
        lb, yb = predict(modelb, eval_windows(held, args.per_span), store, device)
        same_drones = [s for s in block_spans(recs, names, "test", parts=(1,))]
        lt, yt = predict(modelb, eval_windows(same_drones, args.per_span), store, device)
        b = {
            "held_out_recording_acc": (lb.argmax(1) == yb).float().mean().item(),
            "same_drones_time_block_acc": (lt.argmax(1) == yt).float().mean().item(),
            "per_recording": {s.rec.path.name: round((p == s.label).float().mean().item(), 3)
                              for s in held
                              for p in [predict(modelb, eval_windows([s], args.per_span), store, device)[0].argmax(1)]},
        }
        out["runs"].append({"seed": seed, "time_block": a, "cross_recording": b, "seconds": round(time.time() - t0)})
        print(f"seed {seed}: time-block val {a['val']:.3f} test {a['test']:.3f} | cross-recording {b['held_out_recording_acc']:.3f} "
              f"(same drones, unseen block of part 1: {b['same_drones_time_block_acc']:.3f}) [{time.time() - t0:.0f}s]", flush=True)
        for k, v in b["per_recording"].items():
            print(f"    {k}: {v}", flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "closed_set.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
