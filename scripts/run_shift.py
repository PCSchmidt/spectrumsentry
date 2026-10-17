"""Simulated shift sweeps: how fast does a clean-trained model degrade, and what does augmentation recover?

    python scripts/run_shift.py --seeds 3

Models (all trained only on windows that pass the energy detector, time-block train split, 1,200 steps):
  clean      no impairments during training
  aug_all    random impairments per batch (multipath, CFO, noise, receiver bandwidth, I/Q imbalance, ADC bits)
  aug_no_rx  the same but WITHOUT receiver bandwidth limiting and ADC re-quantisation: tests whether
             augmentation generalises to impairments it never saw
Test: the detected windows of the held-out time blocks, with ONE impairment swept at a time.
Metric: macro accuracy (mean of per-drone recall), because the sparse hoppers contribute few windows.
All impairments are simulated stand-ins applied to chamber recordings (docs/design.md); the stated
SNR counts the recording's own noise floor as signal, so true SNR is lower.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from spectrumsentry import augment
from spectrumsentry.catalog import catalog
from spectrumsentry.data import Store, block_spans, class_names, eval_windows, load_batch
from spectrumsentry.gate import Gate, noise_floor, passing_starts
from spectrumsentry.train import predict, train

ROOT = Path(__file__).resolve().parents[1]

SWEEPS = {
    "snr_db": [("clean", {})] + [(str(v), {"snr": float(v)}) for v in (30, 20, 10, 5, 0, -5, -10)],
    "cfo_max_hz": [("0", {})] + [(f"{v:g}", {"cfo": float(v)}) for v in (5e4, 2e5, 1e6, 5e6)],
    "iq_imbalance_gain_db/phase_deg": [("0/0", {})] + [(f"{g}/{p}", {"iq": (g, p)}) for g, p in ((0.5, 3), (1, 8), (2, 15), (3, 25))],
    "multipath_decay/spacing": [("none", {})] + [(f"{d}/{s}", {"mp": (d, s)}) for d, s in ((0.2, 3), (0.5, 3), (0.5, 8), (0.8, 8))],
    "receiver_bandwidth_mhz": [("full", {})] + [(f"{b}", {"bw": b * 1e6}) for b in (56, 20, 10, 5)],
    "adc_bits": [("16 (native)", {})] + [(str(b), {"bits": b}) for b in (12, 8, 6, 4)],
}
MODELS = {"clean": None, "aug_all": True, "aug_no_rx": False}


def macro_accuracy(logits: torch.Tensor, labels: torch.Tensor, n_classes: int) -> float:
    pred = logits.argmax(1)
    recalls = [(pred[labels == c] == c).float().mean().item() for c in range(n_classes) if (labels == c).any()]
    return float(np.mean(recalls))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1200)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    recs = catalog(ROOT / "data" / "cardrf")
    names = class_names(recs)
    store = Store()
    rng = np.random.default_rng(0)
    gate = Gate({r.path.name: noise_floor(r, store, rng) for r in recs}, 0.3, 6.0)

    train_spans = block_spans(recs, names, "train")
    cand = passing_starts(train_spans, store, gate)
    items = eval_windows(block_spans(recs, names, "test"), 80)
    keep = []
    for i in range(0, len(items), 64):
        chunk = items[i : i + 64]
        x, _ = load_batch(store, chunk, "cpu")
        keep += gate.mask(x, chunk).tolist()
    items = [it for it, ok in zip(items, keep) if ok]
    counts = np.bincount([s.label for s, _ in items], minlength=len(names))
    print(f"{len(items)} detected test windows; per drone: {dict(zip(names, counts.tolist()))}", flush=True)

    results: dict = {"classes": names, "test_windows_per_drone": dict(zip(names, counts.tolist())), "steps": args.steps, "sweeps": {}}
    for sweep, levels in SWEEPS.items():
        results["sweeps"][sweep] = {label: {m: [] for m in MODELS} for label, _ in levels}
    for seed in range(args.seeds):
        for mname, receiver in MODELS.items():
            t0 = time.time()
            aug = None if receiver is None else augment.RandomAugment(receiver=receiver, seed=seed)
            model = train(train_spans, len(names), store, device, steps=args.steps, seed=seed, transform=aug, candidates=cand)
            for sweep, levels in SWEEPS.items():
                for label, spec in levels:
                    logits, labels = predict(model, items, store, device, transform=augment.fixed(spec, seed=1000 + seed) if spec else None)
                    results["sweeps"][sweep][label][mname].append(macro_accuracy(logits, labels, len(names)))
            print(f"seed {seed} {mname}: trained+swept in {time.time() - t0:.0f}s; clean macro-acc {results['sweeps']['snr_db']['clean'][mname][-1]:.3f}", flush=True)
        (ROOT / "results" / "shift.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    for sweep, levels in results["sweeps"].items():
        print(f"\n{sweep}")
        for label, by_model in levels.items():
            print(f"  {label:>14}  " + "  ".join(f"{m}: {100 * np.mean(v):5.1f}" for m, v in by_model.items()))


if __name__ == "__main__":
    main()
