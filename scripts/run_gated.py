"""Does an energy-detector gate fix the empty-window errors?

    python scripts/run_gated.py --seeds 3

Time-block split as in run_closed_set.py. Two models per seed:
  ungated: trained on random windows (as before);
  gated:   trained only on windows that pass the energy detector (>= 30% of frames 6 dB above the
           recording's noise floor).
Both are scored on the *detected* test windows, and the detector's per-drone detection rate is
reported, so the end-to-end figure (detected AND correct, as a share of all test windows) is visible.
The gate's noise floor is estimated from the recording itself, which a live receiver could only
approximate from past data; see docs/results.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from spectrumsentry.catalog import catalog
from spectrumsentry.data import Store, block_spans, class_names, eval_windows, load_batch
from spectrumsentry.gate import Gate, noise_floor, passing_starts
from spectrumsentry.train import predict, train

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    recs = catalog(ROOT / "data" / "cardrf")
    names = class_names(recs)
    store = Store()
    rng = np.random.default_rng(0)
    floors = {r.path.name: noise_floor(r, store, rng) for r in recs}
    gate = Gate(floors, min_occupancy=0.3, margin_db=6.0)

    train_spans = block_spans(recs, names, "train")
    cand = passing_starts(train_spans, store, gate)
    print("training windows passing the gate:", {s.rec.path.name[:24]: f"{len(v)}/{s.n_windows}" for s, v in cand.items()}, flush=True)

    test_items = eval_windows(block_spans(recs, names, "test"), 10_000)  # every aligned test window
    passing = []
    for i in range(0, len(test_items), 64):
        chunk = test_items[i : i + 64]
        x, _ = load_batch(store, chunk, "cpu")
        passing += gate.mask(x, chunk).tolist()
    passing = np.array(passing)
    labels_all = np.array([s.label for s, _ in test_items])
    detect = {n: float(passing[labels_all == i].mean()) for i, n in enumerate(names)}
    print("detection rate per drone:", {k: round(v, 2) for k, v in detect.items()}, flush=True)
    gated_items = [it for it, ok in zip(test_items, passing) if ok]

    out = {"classes": names, "detection_rate": detect, "n_test_windows": len(test_items), "n_detected": int(passing.sum()), "runs": []}
    for seed in range(args.seeds):
        row = {"seed": seed}
        for name, kwargs in (("ungated", {}), ("gated", {"candidates": cand})):
            model = train(train_spans, len(names), store, device, steps=args.steps, seed=seed, **kwargs)
            logits, labels = predict(model, test_items, store, device)
            correct = (logits.argmax(1) == labels).numpy()
            row[name] = {
                "accuracy_all_windows": float(correct.mean()),
                "accuracy_detected_windows": float(correct[passing].mean()),
                "end_to_end": float((correct & passing).mean()),
                "per_drone_detected": {n: round(float(correct[(labels.numpy() == i) & passing].mean()), 3) if ((labels.numpy() == i) & passing).any() else None for i, n in enumerate(names)},
            }
        out["runs"].append(row)
        print(f"seed {seed}: " + " | ".join(f"{k}: all {row[k]['accuracy_all_windows']:.3f}, detected {row[k]['accuracy_detected_windows']:.3f}, end-to-end {row[k]['end_to_end']:.3f}" for k in ("ungated", "gated")), flush=True)
    (ROOT / "results" / "gated.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
