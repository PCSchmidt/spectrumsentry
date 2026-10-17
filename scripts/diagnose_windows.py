"""Is the closed-set error caused by windows with no signal in them?

    python scripts/diagnose_windows.py

Trains one model (seed 0, time-block split) and, for every test window, measures its
*occupancy*: the share of its 256-sample frames whose power exceeds the recording's own noise
floor (10th percentile of frame power) by 6 dB. Then reports accuracy for sparse and busy windows,
and per drone.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from spectrumsentry.catalog import catalog
from spectrumsentry.data import WINDOW, Store, block_spans, class_names, eval_windows, load_batch
from spectrumsentry.train import predict, train

ROOT = Path(__file__).resolve().parents[1]
NFFT = 256


def frame_power(x: torch.Tensor) -> torch.Tensor:
    b, n = x.shape
    return x[:, : n // NFFT * NFFT].reshape(b, -1, NFFT).abs().square().mean(-1)  # (B, frames)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    recs = catalog(ROOT / "data" / "cardrf")
    names = class_names(recs)
    store = Store()
    model = train(block_spans(recs, names, "train"), len(names), store, device, steps=600, seed=0)

    # noise floor of each recording: 10th percentile of frame power over 200 random windows
    rng = np.random.default_rng(0)
    floor = {}
    for r in recs:
        starts = rng.integers(0, r.n_samples - WINDOW, 200)
        from spectrumsentry.iq import windows

        floor[r.path.name] = float(np.percentile(frame_power(windows(store.iq(r), starts, WINDOW)).flatten().numpy(), 10))

    items = eval_windows(block_spans(recs, names, "test"), 60)
    logits, labels = predict(model, items, store, device)
    correct = (logits.argmax(1) == labels).numpy()
    occupancy = []
    for i in range(0, len(items), 48):
        x, _ = load_batch(store, items[i : i + 48], "cpu")
        fp = frame_power(x)
        for (s, _), row in zip(items[i : i + 48], fp):
            occupancy.append(float((row > floor[s.rec.path.name] * 10 ** 0.6).float().mean()))
    occupancy = np.array(occupancy)
    out = {"overall_accuracy": float(correct.mean()), "bins": [], "per_drone": {}}
    for lo, hi in ((0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.01)):
        m = (occupancy >= lo) & (occupancy < hi)
        out["bins"].append({"occupancy": f"{lo:.1f}-{min(hi, 1):.1f}", "windows": int(m.sum()), "accuracy": float(correct[m].mean()) if m.any() else None})
    for i, n in enumerate(names):
        m = labels.numpy() == i
        out["per_drone"][n] = {"mean_occupancy": round(float(occupancy[m].mean()), 3), "accuracy": round(float(correct[m].mean()), 3)}
    (ROOT / "results" / "diagnose_windows.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
