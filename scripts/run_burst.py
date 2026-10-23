"""Which drones are confused, and do longer observations fix the sparse hoppers?

    python scripts/run_burst.py --seeds 3

Same time-block split and energy-detector gate as run_gated.py; the classifier is the gated model, three seeds.
Part 1: a confusion matrix on the detected test windows, summed over seeds.
Part 2: segment-level decisions. Consecutive aligned test windows of one recording are grouped into segments of n
windows (0.55 ms each). A segment is *detected* if any of its windows passes the gate, and is classified by summing
the log-probabilities of its passing windows. For each n: the share of segments detected, the accuracy among
detected segments, and the share detected and correct, per drone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from spectrumsentry.catalog import catalog
from spectrumsentry.data import WINDOW, Store, block_spans, class_names, eval_windows, load_batch
from spectrumsentry.gate import Gate, noise_floor, passing_starts
from spectrumsentry.train import predict, train

ROOT = Path(__file__).resolve().parents[1]
NS = [1, 5, 20, 100, 400]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    recs = catalog(ROOT / "data" / "cardrf")
    names = class_names(recs)
    k = len(names)
    store = Store()
    rng = np.random.default_rng(0)
    floors = {r.path.name: noise_floor(r, store, rng) for r in recs}
    gate = Gate(floors, min_occupancy=0.3, margin_db=6.0)
    train_spans = block_spans(recs, names, "train")
    cand = passing_starts(train_spans, store, gate)

    # all aligned test windows (every span has fewer than 10,000, so none are skipped and consecutive ones stay adjacent)
    test_items = eval_windows(block_spans(recs, names, "test"), 10**9)
    passing = []
    for i in range(0, len(test_items), 64):
        chunk = test_items[i : i + 64]
        x, _ = load_batch(store, chunk, "cpu")
        passing += gate.mask(x, chunk).tolist()
    passing = np.array(passing)
    span_id = np.array([id(s) for s, _ in test_items])
    starts = np.array([st for _, st in test_items])
    assert all(starts[i + 1] - starts[i] == WINDOW for i in range(len(starts) - 1) if span_id[i + 1] == span_id[i]), "windows are not contiguous"
    labels = np.array([s.label for s, _ in test_items])

    conf = np.zeros((k, k), dtype=int)
    seg_stats = {n: {"segments": np.zeros(k), "detected": np.zeros(k), "correct": np.zeros(k)} for n in NS}
    for seed in range(args.seeds):
        model = train(train_spans, k, store, device, steps=args.steps, seed=seed, candidates=cand)
        logits, lab = predict(model, test_items, store, device)
        logp = torch.log_softmax(logits, dim=1).numpy()
        pred = logits.argmax(1).numpy()
        for t, p in zip(labels[passing], pred[passing]):
            conf[t, p] += 1
        # segments
        for n in NS:
            i = 0
            while i < len(test_items):
                j = i
                while j < len(test_items) and span_id[j] == span_id[i]:
                    j += 1
                for a in range(i, j - n + 1, n):
                    sel = np.arange(a, a + n)
                    ok = passing[sel]
                    lbl = labels[a]
                    seg_stats[n]["segments"][lbl] += 1
                    if ok.any():
                        seg_stats[n]["detected"][lbl] += 1
                        if logp[sel[ok]].sum(axis=0).argmax() == lbl:
                            seg_stats[n]["correct"][lbl] += 1
                i = j
        print(f"seed {seed} done", flush=True)

    out = {"classes": names, "confusion_counts": conf.tolist(), "segment_windows": NS, "per_drone": {}}
    short = [n[:18] for n in names]
    print("\nConfusion matrix on detected test windows (rows: true drone, columns: predicted; % of the row; seeds pooled)")
    print(" " * 20 + " ".join(f"{s[:6]:>6}" for s in short))
    for i, n in enumerate(names):
        row = conf[i] / max(conf[i].sum(), 1)
        print(f"{n[:19]:<20}" + " ".join(f"{100 * v:6.1f}" for v in row), f" (n={conf[i].sum()})")
    pairs = sorted(((conf[i, j] / max(conf[i].sum(), 1), names[i], names[j], int(conf[i].sum())) for i in range(k) for j in range(k) if i != j), reverse=True)[:8]
    print("\nLargest confusions (share of the true drone's detected windows sent to the other drone):")
    for v, a, b, n in pairs:
        print(f"  {a} -> {b}: {100 * v:.1f}% (of {n})")
    out["top_confusions"] = [{"true": a, "predicted": b, "share": v, "n": n} for v, a, b, n in pairs]

    print("\nSegment-level decisions (n windows of 0.55 ms): detected share / accuracy among detected / detected-and-correct share")
    print(f"{'drone':<22}" + "".join(f"{'n=' + str(n) + ' (' + format(n * 0.55, '.1f') + ' ms)':^27}" for n in NS))
    for i, name in enumerate(names):
        cells = []
        out["per_drone"][name] = {}
        for n in NS:
            s = seg_stats[n]
            seg, det, cor = s["segments"][i], s["detected"][i], s["correct"][i]
            cells.append(f"{100 * det / seg:5.0f}% {100 * cor / max(det, 1):5.0f}% {100 * cor / seg:5.0f}%   " if seg else f"{'n/a':^27}")
            out["per_drone"][name][str(n)] = {"segments": int(seg), "detected": int(det), "correct": int(cor)}
        print(f"{name[:21]:<22}" + "".join(f"{c:^27}" for c in cells))
    (ROOT / "results" / "burst.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
