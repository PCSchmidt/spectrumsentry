"""DroneRF: closed-set baselines and a noise sweep, as an independent check on the CardRF findings.

    python scripts/run_dronerf.py --seeds 3

Splits hold out whole 0.25 s segments (per class: first 60% train, next 10% validation, last 30% test).
Tasks: "drone" (background / AR / Bebop / Phantom) and "bui" (the 10 codes: drone and flight mode).
Noise sweep on the drone task: a model trained clean against one trained with random noise (SNR -5..30 dB).
Everything here is a different team, place and receivers from CardRF, but the class sets differ, so no model
is transferred between the datasets; only the *method and its fragility* are compared.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from spectrumsentry import dronerf as D

ROOT = Path(__file__).resolve().parents[1]
SNRS = [None, 30, 20, 10, 5, 0, -5, -10]


def macro(logits, labels, k):
    pred = logits.argmax(1)
    return float(np.mean([(pred[labels == c] == c).float().mean().item() for c in range(k) if (labels == c).any()]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--split-mode", choices=["ordered", "interleaved"], default="ordered")
    ap.add_argument("--tasks-only", action="store_true", help="skip the noise sweep")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segs = D.catalog(ROOT / "data" / "dronerf" / "npy")
    print(f"{len(segs)} segments", flush=True)
    store = D.Store()
    out: dict = {"n_segments": len(segs), "steps": args.steps, "tasks": {}, "noise_sweep": {}}

    for task in ("drone", "bui"):
        classes = sorted({D.label_of(s, task) for s in segs})
        parts = D.split(segs, task, args.split_mode)
        out["tasks"][task] = {"classes": classes, "n_segments": {k: len(v) for k, v in parts.items()}, "runs": []}
        for seed in range(args.seeds):
            t0 = time.time()
            model = D.train(parts["train"], classes, task, store, device, args.steps, seed=seed)
            row = {"seed": seed}
            for split in ("val", "test"):
                lg, lb = D.predict(model, D.eval_items(parts[split], classes, task), store, device)
                row[split] = {"accuracy": (lg.argmax(1) == lb).float().mean().item(), "macro": macro(lg, lb, len(classes)),
                              "per_class": {c: round((lg.argmax(1)[lb == i] == i).float().mean().item(), 3) for i, c in enumerate(classes) if (lb == i).any()}}
            out["tasks"][task]["runs"].append(row)
            print(f"{task} seed {seed}: val {row['val']['accuracy']:.3f} test {row['test']['accuracy']:.3f} macro {row['test']['macro']:.3f} [{time.time() - t0:.0f}s]", flush=True)

    if args.tasks_only:
        suffix = "" if args.split_mode == "ordered" else f"_{args.split_mode}"
        (ROOT / "results" / f"dronerf_tasks{suffix}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        return
    # noise sweep, drone task
    task = "drone"
    classes = sorted({D.label_of(s, task) for s in segs})
    parts = D.split(segs, task, args.split_mode)
    items = D.eval_items(parts["test"], classes, task)
    rows = {str(s): {"clean": [], "noise_aug": []} for s in SNRS}
    for seed in range(args.seeds):
        gen_aug = torch.Generator(device=device).manual_seed(seed)

        def aug(x, g=gen_aug):
            if torch.rand(1, generator=g, device=x.device).item() < 0.8:
                snr = -5 + 35 * torch.rand(1, generator=g, device=x.device).item()
                return D.awgn_real(x, snr, g)
            return x

        models = {"clean": D.train(parts["train"], classes, task, store, device, args.steps, seed=seed),
                  "noise_aug": D.train(parts["train"], classes, task, store, device, args.steps, seed=seed, transform=aug)}
        for snr in SNRS:
            for name, model in models.items():
                g = torch.Generator(device=device).manual_seed(1000 + seed)
                tf = None if snr is None else (lambda x, s=snr, g=g: D.awgn_real(x, float(s), g))
                lg, lb = D.predict(model, items, store, device, transform=tf)
                rows[str(snr)][name].append(macro(lg, lb, len(classes)))
        print(f"noise sweep seed {seed} done", flush=True)
    out["noise_sweep"] = {"classes": classes, "macro_accuracy_by_snr": rows}
    (ROOT / "results" / "dronerf.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nnoise sweep (macro accuracy, drone task)")
    for snr, r in rows.items():
        print(f"  SNR {snr:>5}: clean {100 * np.mean(r['clean']):5.1f}   noise-aug {100 * np.mean(r['noise_aug']):5.1f}")


if __name__ == "__main__":
    main()
