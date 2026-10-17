"""Read interleaved int16 IQ recordings and cut fixed-length windows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_int16(path: Path) -> np.ndarray:
    """Whole recording as an (N, 2) int16 array (I, Q), memory-mapped (not read until used)."""
    raw = np.memmap(path, dtype="<i2", mode="r")
    return raw[: len(raw) // 2 * 2].reshape(-1, 2)


def windows(iq: np.ndarray, starts: np.ndarray, length: int) -> torch.Tensor:
    """Stack windows ``iq[s:s+length]`` into a complex64 tensor (B, length), scaled to roughly unit range."""
    batch = np.stack([np.asarray(iq[s : s + length], dtype=np.float32) for s in starts])  # (B, length, 2)
    t = torch.from_numpy(batch) / 32768.0
    return torch.complex(t[..., 0], t[..., 1])
