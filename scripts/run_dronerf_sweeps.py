"""DroneRF: bandwidth and bit-depth sweeps (the impairments that apply to a real-valued recording).

    python scripts/run_dronerf_sweeps.py --seeds 3

Drone task (background / AR / Bebop / Phantom), ordered split as in run_dronerf.py. Two models per seed: one trained
clean and one trained with random noise (SNR -5..30 dB), random band-limiting (keep 6%-100% of the spectrum) and
random quantisation (3-12 bits), each applied with probability 0.5. Both are tested clean and under each sweep
(bandwidth kept: 100%, 50%, 25%, 12.5%, 6.25%; ADC bits: 12, 8, 6, 4, 3, 2), macro accuracy over the 4 classes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from spectrumsentry import dronerf as D

ROOT = Path(__file__).resolve().parents[1]
KEEP = [1.0, 0.5, 0.25, 0.125, 0.0625]
BITS = [12, 8, 6, 4, 3, 2]


def macro(logits, labels, k):
    pred = logits.argmax(1)
    return float(np.mean([(pred[labels == c] == c).float().mean().item() for c in range(k) if (labels == c).any()]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1200)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    segs = D.catalog(ROOT / "data" / "dronerf" / "npy")
    store = D.Store()
    task = "drone"
    classes = sorted({D.label_of(s, task) for s in segs})
    parts = D.split(segs, task, "ordered")
    items = D.eval_items(parts["test"], classes, task)
    rows = {"clean_test": {"clean": [], "aug": []}, "keep": {str(k): {"clean": [], "aug": []} for k in KEEP}, "bits": {str(b): {"clean": [], "aug": []} for b in BITS}}
    for seed in range(args.seeds):
        gen = torch.Generator(device=device).manual_seed(seed)

        def aug(x, g=gen):
            if torch.rand(1, generator=g, device=x.device).item() < 0.5:
                x = D.awgn_real(x, -5 + 35 * torch.rand(1, generator=g, device=x.device).item(), g)
            if torch.rand(1, generator=g, device=x.device).item() < 0.5:
                x = D.bandlimit_real(x, 0.0625 + 0.9375 * torch.rand(1, generator=g, device=x.device).item())
            if torch.rand(1, generator=g, device=x.device).item() < 0.5:
                x = D.quantize_real(x, int(3 + torch.randint(0, 10, (1,), generator=g, device=x.device).item()))
            return x

        models = {"clean": D.train(parts["train"], classes, task, store, device, args.steps, seed=seed),
                  "aug": D.train(parts["train"], classes, task, store, device, args.steps, seed=seed, transform=aug)}
        for name, model in models.items():
            lg, lb = D.predict(model, items, store, device)
            rows["clean_test"][name].append(macro(lg, lb, len(classes)))
            for kf in KEEP:
                lg, lb = D.predict(model, items, store, device, transform=lambda x, kf=kf: D.bandlimit_real(x, kf))
                rows["keep"][str(kf)][name].append(macro(lg, lb, len(classes)))
            for b in BITS:
                lg, lb = D.predict(model, items, store, device, transform=lambda x, b=b: D.quantize_real(x, b))
                rows["bits"][str(b)][name].append(macro(lg, lb, len(classes)))
        print(f"seed {seed} done", flush=True)
    out = {"classes": classes, "macro_accuracy": rows}
    (ROOT / "results" / "dronerf_sweeps.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    m = lambda v: f"{100 * np.mean(v):5.1f} +- {100 * np.std(v):4.1f}"  # noqa: E731
    print(f"\nclean test: clean-trained {m(rows['clean_test']['clean'])}   augmented {m(rows['clean_test']['aug'])}")
    print("\nbandwidth kept (fraction of the spectrum):")
    for kf in KEEP:
        r = rows["keep"][str(kf)]
        print(f"  {kf:>6}: clean-trained {m(r['clean'])}   augmented {m(r['aug'])}")
    print("\nADC bits:")
    for b in BITS:
        r = rows["bits"][str(b)]
        print(f"  {b:>6}: clean-trained {m(r['clean'])}   augmented {m(r['aug'])}")


if __name__ == "__main__":
    main()
