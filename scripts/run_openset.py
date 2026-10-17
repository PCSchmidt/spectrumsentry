"""Open-set recognition on CardRF: can the model tell that a window is from a drone it has never seen?

    python scripts/run_openset.py --seeds 3

The 10 drone models are shuffled once (seed 0) into 5 pairs. In each fold one pair is held out as
"unknown": never used in training, so the model is a closed-set classifier over the other 8. Test windows
are the *detected* windows (energy-detector gate) of the held-out time blocks, for known and unknown drones
alike. Three post-hoc scores (MSP, energy, Mahalanobis) are compared by AUROC for known-versus-unknown, and
by the share of unknown windows accepted when 95% of known windows are accepted.

Windows are correlated within a recording, so the spread reported is across folds and seeds, not a
confidence interval. Two unknown drones per fold means some folds are much harder than others (an unknown
DJI model next to a known DJI sibling is the hard case); the per-fold table shows it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from spectrumsentry.catalog import catalog
from spectrumsentry.data import Span, Store, block_spans, class_names, eval_windows, load_batch, sample_from_candidates
from spectrumsentry.gate import Gate, noise_floor, passing_starts
from spectrumsentry.openset import Mahalanobis, auroc, energy, msp, unknown_accepted_at_tpr
from spectrumsentry.spec import spectrogram
from spectrumsentry.train import train

ROOT = Path(__file__).resolve().parents[1]
SCORES = ("msp", "energy", "maha")


@torch.no_grad()
def forward_all(model, items, store, device, chunk=48):
    model.eval()
    logits, feats, labels = [], [], []
    for i in range(0, len(items), chunk):
        x, y = load_batch(store, items[i : i + chunk], device)
        s = spectrogram(x)
        feats.append(model.embed(s).cpu())
        logits.append(model(s).cpu())
        labels.append(y.cpu())
    return torch.cat(logits), torch.cat(feats), torch.cat(labels)


def detected(items, gate, store):
    keep = []
    for i in range(0, len(items), 64):
        chunk = items[i : i + 64]
        x, _ = load_batch(store, chunk, "cpu")
        keep += gate.mask(x, chunk).tolist()
    return [it for it, ok in zip(items, keep) if ok]


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
    order = np.random.default_rng(0).permutation(len(names))
    folds = [[names[i] for i in order[2 * k : 2 * k + 2]] for k in range(5)]
    print("folds (unknown pairs):", folds, flush=True)

    train_all = block_spans(recs, names, "train")
    test_all = block_spans(recs, names, "test")
    out = {"folds": folds, "steps": args.steps, "runs": []}
    for f, unknown in enumerate(folds):
        known = [n for n in names if n not in unknown]
        remap = {names.index(n): i for i, n in enumerate(known)}
        tr = [Span(s.rec, s.lo, s.hi, remap[s.label]) for s in train_all if s.label in remap]
        kn_test = [Span(s.rec, s.lo, s.hi, remap[s.label]) for s in test_all if s.label in remap]
        un_test = [Span(s.rec, s.lo, s.hi, -1) for s in test_all if s.label not in remap]
        cand = passing_starts(tr, store, gate)
        kn_items = detected(eval_windows(kn_test, 80), gate, store)
        un_items = detected(eval_windows(un_test, 80), gate, store)
        un_class = np.array([names.index(s.rec.drone) for s, _ in un_items])
        for seed in range(args.seeds):
            model = train(tr, len(known), store, device, steps=args.steps, seed=seed, candidates=cand)
            fit_items = sample_from_candidates(tr, cand, 2000, np.random.default_rng(seed))
            fl, ff, fy = forward_all(model, fit_items, store, device)
            maha = Mahalanobis(ff, fy, len(known))
            kl, kf, ky = forward_all(model, kn_items, store, device)
            ul, uf, _ = forward_all(model, un_items, store, device)
            known_scores = {"msp": msp(kl), "energy": energy(kl), "maha": maha.score(kf)}
            unknown_scores = {"msp": msp(ul), "energy": energy(ul), "maha": maha.score(uf)}
            row = {"fold": f, "seed": seed, "unknown": unknown, "n_known_windows": len(kn_items), "n_unknown_windows": len(un_items),
                   "closed_set_accuracy_known": (kl.argmax(1) == ky).float().mean().item(), "scores": {}}
            for sc in SCORES:
                thr, acc_unk = unknown_accepted_at_tpr(known_scores[sc], unknown_scores[sc], 0.95)
                accepted = known_scores[sc] >= thr
                per_unknown = {}
                for u in unknown:
                    m = torch.tensor(un_class == names.index(u))
                    per_unknown[u] = {"windows": int(m.sum()), "auroc": auroc(known_scores[sc], unknown_scores[sc][m]) if m.any() else None}
                row["scores"][sc] = {
                    "auroc": auroc(known_scores[sc], unknown_scores[sc]),
                    "unknown_accepted_at_95pct_known": acc_unk,
                    "closed_set_accuracy_on_accepted_known": (kl.argmax(1)[accepted] == ky[accepted]).float().mean().item(),
                    "per_unknown": per_unknown,
                }
            out["runs"].append(row)
            print(f"fold {f} {unknown} seed {seed}: closed-set {row['closed_set_accuracy_known']:.3f} | "
                  + " | ".join(f"{sc} AUROC {row['scores'][sc]['auroc']:.3f} unk-acc@95 {row['scores'][sc]['unknown_accepted_at_95pct_known']:.2f}" for sc in SCORES), flush=True)
        (ROOT / "results" / "openset.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\nsummary over folds x seeds")
    for sc in SCORES:
        a = [r["scores"][sc]["auroc"] for r in out["runs"]]
        u = [r["scores"][sc]["unknown_accepted_at_95pct_known"] for r in out["runs"]]
        print(f"  {sc:7s} AUROC mean {np.mean(a):.3f} (min {min(a):.3f}, max {max(a):.3f}); unknown accepted at 95% known: mean {np.mean(u):.2f}")


if __name__ == "__main__":
    main()
