"""The leaky contrast: what a random split of windows would have reported.

    python scripts/run_leaky.py --seeds 3

Aligned windows of every recording are shuffled 80/20 into train and test, ignoring time. Neighbouring
windows of one recording are near-duplicates, so this measures memorisation of the recording, not
recognition of the drone. Reported only as a contrast to the time-block split in run_closed_set.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from spectrumsentry.catalog import catalog
from spectrumsentry.data import WINDOW, Span, Store, class_names, eval_windows
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
    accs = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(100 + seed)
        # one single-window Span per aligned window: a random 80/20 split of windows, per recording
        train_spans, test_items = [], []
        for r in recs:
            n_win = r.n_samples // WINDOW
            order = rng.permutation(n_win)
            cut = int(0.8 * n_win)
            label = names.index(r.drone)
            train_spans += [Span(r, int(i) * WINDOW, int(i + 1) * WINDOW, label) for i in order[:cut]]
            test_items += [(Span(r, int(i) * WINDOW, int(i + 1) * WINDOW, label), int(i) * WINDOW) for i in order[cut:][:60]]
        model = train(train_spans, len(names), store, device, steps=args.steps, seed=seed)
        logits, labels = predict(model, test_items, store, device)
        accs.append((logits.argmax(1) == labels).float().mean().item())
        print(f"seed {seed}: random-window split test accuracy {accs[-1]:.3f} ({len(test_items)} windows)", flush=True)
    (ROOT / "results" / "leaky.json").write_text(json.dumps({"random_window_split_test_accuracy": accs}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
