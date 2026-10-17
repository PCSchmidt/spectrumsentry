"""DroneRF (Al-Sa'd et al., Qatar University, CC BY 4.0): segments, splits, windows and training.

Each recording segment (0.25 s, 10,000,000 real-valued samples per half) comes as an "L" and an "H"
file, the lower and upper half of the 2.4 GHz band captured by two receivers. The sample rate is not
stated by the dataset page; 40 MS/s is inferred from 10.25 s per background record in 41 segments.
The two halves are stacked as two input channels. Splits hold out whole segments.

File name code (five bits, from the paper, confirmed by the folders): 00000 background;
10000-10011 Bebop; 10100-10111 AR; 11000 Phantom; the last two bits are the flight mode.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .model import SpecCnn

WINDOW = 65_536  # real samples, 1.6 ms at the inferred 40 MS/s
DRONE_OF = {"000": "background", "100": "bebop", "101": "ar", "110": "phantom"}
_NAME = re.compile(r"^(\d{5})([LH])_(\d+)\.npy$")


@dataclass(frozen=True)
class Seg:
    bui: str  # five-bit code
    idx: int  # segment number
    low: Path
    high: Path

    @property
    def drone(self) -> str:
        return DRONE_OF[self.bui[:3]]


def catalog(npy_dir: Path) -> list[Seg]:
    halves: dict[tuple[str, int], dict[str, Path]] = {}
    for p in npy_dir.glob("*.npy"):
        m = _NAME.match(p.name)
        if m:
            halves.setdefault((m[1], int(m[3])), {})[m[2]] = p
    return sorted(
        (Seg(b, i, h["L"], h["H"]) for (b, i), h in halves.items() if "L" in h and "H" in h),
        key=lambda s: (s.bui, s.idx),
    )


def label_of(seg: Seg, task: str) -> str:
    return seg.drone if task == "drone" else seg.bui


def split(segs: list[Seg], task: str, mode: str = "ordered") -> dict[str, list[Seg]]:
    """Whole-segment splits.

    ``ordered``: per class, by segment number, first 60% train, next 10% validation, last 30% test, so
    test segments come later in the recording than training ones (tests drift over the session).
    ``interleaved``: segments are dealt out round-robin (roughly 60/10/30) regardless of time, so train and
    test segments are neighbours in time (no drift between them, but still no shared segment).
    """
    out: dict[str, list[Seg]] = {"train": [], "val": [], "test": []}
    by_bui: dict[str, list[Seg]] = {}
    for s in segs:
        by_bui.setdefault(s.bui, []).append(s)
    for group in by_bui.values():
        group.sort(key=lambda s: s.idx)
        n = len(group)
        if mode == "ordered":
            a, b = int(round(0.6 * n)), int(round(0.7 * n))
            out["train"] += group[:a]
            out["val"] += group[a:b]
            out["test"] += group[b:]
        elif mode == "interleaved":
            # pattern over 10 consecutive segments: 6 train, 1 val, 3 test, spread so test segments sit among train ones
            pattern = ["train", "train", "test", "train", "train", "test", "train", "val", "test", "train"]
            for i, seg in enumerate(group):
                out[pattern[i % 10]].append(seg)
        else:
            raise ValueError(mode)
    return out


class Store:
    def __init__(self):
        self._open: dict[Path, np.ndarray] = {}

    def get(self, p: Path) -> np.ndarray:
        if p not in self._open:
            self._open[p] = np.load(p, mmap_mode="r")
        return self._open[p]


def window(store: Store, seg: Seg, start: int) -> torch.Tensor:
    """(2, WINDOW) float32: low and high half, scaled by 1/8192 (values are 14-bit)."""
    return torch.from_numpy(np.stack([np.asarray(store.get(seg.low)[start : start + WINDOW], dtype=np.float32), np.asarray(store.get(seg.high)[start : start + WINDOW], dtype=np.float32)])) / 8192.0


def spectrogram(x: torch.Tensor, nfft: int = 256, out: int = 64) -> torch.Tensor:
    """(B, 2, N) real to (B, 2, out, out): log power of real FFT frames, pooled, standardised per example and channel."""
    b, c, n = x.shape
    frames = x[..., : n // nfft * nfft].reshape(b, c, -1, nfft) * torch.hann_window(nfft, device=x.device)
    img = torch.log10(torch.fft.rfft(frames, dim=-1).abs().square() + 1e-12)
    img = F.adaptive_avg_pool2d(img, out)
    mean = img.mean(dim=(2, 3), keepdim=True)
    std = img.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return (img - mean) / std


Transform = Callable[[torch.Tensor], torch.Tensor]


def awgn_real(x: torch.Tensor, snr_db: float, gen: torch.Generator | None = None) -> torch.Tensor:
    """Add white Gaussian noise so each channel's mean power over noise power equals ``snr_db``."""
    p = x.square().mean(dim=-1, keepdim=True)
    return x + torch.sqrt(p / 10 ** (snr_db / 10)) * torch.randn(x.shape, generator=gen, device=x.device)


def bandlimit_real(x: torch.Tensor, keep_fraction: float) -> torch.Tensor:
    """Keep only the lowest ``keep_fraction`` of the spectrum of each real signal (a narrower receiver)."""
    spec = torch.fft.rfft(x, dim=-1)
    cut = int(spec.shape[-1] * keep_fraction)
    spec[..., cut:] = 0
    return torch.fft.irfft(spec, n=x.shape[-1], dim=-1)


def quantize_real(x: torch.Tensor, bits: int) -> torch.Tensor:
    peak = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    levels = 2 ** (bits - 1) - 1
    return torch.round(x / peak * levels) / levels * peak


def sample(segs: list[Seg], classes: list[str], task: str, n: int, rng: np.random.Generator) -> list[tuple[Seg, int, int]]:
    """n (segment, start, label) draws: class uniform, then segment, then a random window start."""
    by_class: dict[int, list[Seg]] = {}
    for s in segs:
        by_class.setdefault(classes.index(label_of(s, task)), []).append(s)
    keys = sorted(by_class)
    out = []
    for c in rng.choice(keys, size=n):
        s = by_class[int(c)][rng.integers(len(by_class[int(c)]))]
        n_samples = len(np.load(s.low, mmap_mode="r"))
        out.append((s, int(rng.integers(0, n_samples - WINDOW + 1)), int(c)))
    return out


def eval_items(segs: list[Seg], classes: list[str], task: str, per_seg: int = 20) -> list[tuple[Seg, int, int]]:
    out = []
    for s in segs:
        n_samples = len(np.load(s.low, mmap_mode="r"))
        n_win = n_samples // WINDOW
        for i in np.unique(np.linspace(0, n_win - 1, per_seg).round().astype(int)):
            out.append((s, int(i) * WINDOW, classes.index(label_of(s, task))))
    return out


def batch(store: Store, items, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.stack([window(store, s, st) for s, st, _ in items]).to(device)
    return x, torch.tensor([c for *_, c in items], device=device)


def train(segs, classes, task, store, device, steps=1200, bsz=48, seed=0, transform: Transform | None = None, lr=2e-3) -> SpecCnn:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SpecCnn(len(classes), in_channels=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    model.train()
    for _ in range(steps):
        x, y = batch(store, sample(segs, classes, task, bsz, rng), device)
        if transform is not None:
            x = transform(x)
        loss = nn.functional.cross_entropy(model(spectrogram(x)), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return model


@torch.no_grad()
def predict(model: SpecCnn, items, store: Store, device: str, transform: Transform | None = None, chunk: int = 48):
    model.eval()
    logits, labels = [], []
    for i in range(0, len(items), chunk):
        x, y = batch(store, items[i : i + chunk], device)
        if transform is not None:
            x = transform(x)
        logits.append(model(spectrogram(x)).cpu())
        labels.append(y.cpu())
    return torch.cat(logits), torch.cat(labels)
