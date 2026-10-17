"""Energy-detector gate: keep only windows that actually contain signal.

Frequency-hopping drone links are silent most of the time, so a fixed-length window often holds
nothing but noise (see docs/results.md). A deployed receiver would run an energy detector in
front of the classifier. ``Gate`` is that detector: a window passes if enough of its short frames
are a few dB above the recording's noise floor.
"""

from __future__ import annotations

import numpy as np
import torch

from .catalog import Recording
from .data import WINDOW, Span, Store, load_batch
from .iq import windows

NFFT = 256


def frame_power(x: torch.Tensor, nfft: int = NFFT) -> torch.Tensor:
    """Mean power of each non-overlapping ``nfft``-sample frame: (B, N) complex to (B, frames)."""
    b, n = x.shape
    return x[:, : n // nfft * nfft].reshape(b, -1, nfft).abs().square().mean(-1)


def noise_floor(rec: Recording, store: Store, rng: np.random.Generator, n_windows: int = 200) -> float:
    """10th percentile of frame power over random windows of the recording (an estimate of its idle level)."""
    starts = rng.integers(0, rec.n_samples - WINDOW, n_windows)
    return float(np.percentile(frame_power(windows(store.iq(rec), starts, WINDOW)).flatten().numpy(), 10))


class Gate:
    def __init__(self, floors: dict[str, float], min_occupancy: float = 0.3, margin_db: float = 6.0):
        self.floors, self.min_occupancy, self.margin = floors, min_occupancy, 10 ** (margin_db / 10)

    def occupancy(self, x: torch.Tensor, items) -> torch.Tensor:
        """Share of each window's frames more than ``margin_db`` above its recording's noise floor."""
        fp = frame_power(x)
        floor = torch.tensor([self.floors[s.rec.path.name] for s, _ in items], device=x.device, dtype=fp.dtype).unsqueeze(1)
        return (fp > floor * self.margin).float().mean(dim=1)

    def mask(self, x: torch.Tensor, items) -> torch.Tensor:
        return self.occupancy(x, items) >= self.min_occupancy


def passing_starts(spans: list[Span], store: Store, gate: Gate, chunk: int = 64) -> dict[Span, np.ndarray]:
    """For each span, the start offsets of its aligned windows that pass the gate (scans the whole span)."""
    out: dict[Span, np.ndarray] = {}
    for s in spans:
        starts = [s.lo + i * WINDOW for i in range(s.n_windows)]
        keep: list[int] = []
        for i in range(0, len(starts), chunk):
            items = [(s, st) for st in starts[i : i + chunk]]
            x, _ = load_batch(store, items, "cpu")
            m = gate.mask(x, items)
            keep += [st for (_, st), ok in zip(items, m) if ok]
        out[s] = np.array(keep, dtype=np.int64)
    return out
