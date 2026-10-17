"""Splits, window sampling and evaluation over the CardRF recordings.

Windows cut from one recording might be near-duplicates of each other (the same hopping
pattern repeats), so a random split of windows could measure memorisation (on this dataset
it measured the same as the time-block split; see docs/results.md). Each
recording is divided into contiguous time blocks, with unused gaps between them:

    train 0-60% | gap | validation 65-75% | gap | test 80-100%

and, separately, drones that were recorded in two files (a "1of2" and a "2of2") give a
cross-recording test: train on part 1, test on part 2. See docs/design.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .catalog import Recording
from .iq import load_int16, windows

WINDOW = 65_536  # samples per example (0.55 ms at 120 MS/s)
BLOCKS = {"train": (0.00, 0.60), "val": (0.65, 0.75), "test": (0.80, 1.00)}


@dataclass(frozen=True)
class Span:
    rec: Recording
    lo: int  # first sample (inclusive)
    hi: int  # last sample (exclusive)
    label: int

    @property
    def n_windows(self) -> int:
        return max(0, (self.hi - self.lo) // WINDOW)


def class_names(recs: list[Recording]) -> list[str]:
    return sorted({r.drone for r in recs})


def block_spans(recs: list[Recording], names: list[str], block: str, parts: tuple[int, ...] | None = None) -> list[Span]:
    """The ``block`` ("train", "val" or "test") span of each recording, optionally only for given part numbers.

    ``parts`` filters two-file drones: e.g. (0, 1) keeps single-file recordings and part 1.
    """
    f0, f1 = BLOCKS[block]
    return [
        Span(r, int(r.n_samples * f0), int(r.n_samples * f1), names.index(r.drone))
        for r in recs
        if names.index(r.drone) >= 0 and (parts is None or r.part in parts)
    ]


def whole_spans(recs: list[Recording], names: list[str], parts: tuple[int, ...]) -> list[Span]:
    """Entire recordings (for the cross-recording test), restricted to the given part numbers."""
    return [Span(r, 0, r.n_samples, names.index(r.drone)) for r in recs if r.part in parts]


class Store:
    """Memory-mapped recordings, opened once."""

    def __init__(self):
        self._open: dict[str, np.ndarray] = {}

    def iq(self, rec: Recording) -> np.ndarray:
        key = str(rec.path)
        if key not in self._open:
            self._open[key] = load_int16(rec.path)
        return self._open[key]


def sample_train(spans: list[Span], n: int, rng: np.random.Generator) -> list[tuple[Span, int]]:
    """``n`` random (span, start) pairs with classes drawn uniformly, then a span of that class, then a random start."""
    by_class: dict[int, list[Span]] = {}
    for s in spans:
        if s.n_windows:
            by_class.setdefault(s.label, []).append(s)
    classes = sorted(by_class)
    out = []
    for c in rng.choice(classes, size=n):
        s = by_class[int(c)][rng.integers(len(by_class[int(c)]))]
        out.append((s, int(rng.integers(s.lo, s.hi - WINDOW + 1))))
    return out


def sample_from_candidates(spans: list[Span], candidates: dict[Span, np.ndarray], n: int, rng: np.random.Generator) -> list[tuple[Span, int]]:
    """Like ``sample_train`` but starts come from ``candidates[span]`` (e.g. windows that passed a gate)."""
    by_class: dict[int, list[Span]] = {}
    for s in spans:
        if len(candidates.get(s, ())):
            by_class.setdefault(s.label, []).append(s)
    classes = sorted(by_class)
    out = []
    for c in rng.choice(classes, size=n):
        s = by_class[int(c)][rng.integers(len(by_class[int(c)]))]
        out.append((s, int(rng.choice(candidates[s]))))
    return out


def eval_windows(spans: list[Span], per_span: int) -> list[tuple[Span, int]]:
    """Up to ``per_span`` evenly spaced, non-overlapping aligned windows from each span."""
    out = []
    for s in spans:
        if not s.n_windows:
            continue
        idx = np.unique(np.linspace(0, s.n_windows - 1, min(per_span, s.n_windows)).round().astype(int))
        out += [(s, s.lo + int(i) * WINDOW) for i in idx]
    return out


def load_batch(store: Store, items: list[tuple[Span, int]], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """IQ windows (B, WINDOW) complex on ``device`` and integer labels (B,)."""
    x = torch.cat([windows(store.iq(s.rec), np.array([start]), WINDOW) for s, start in items])
    y = torch.tensor([s.label for s, _ in items])
    return x.to(device), y.to(device)
