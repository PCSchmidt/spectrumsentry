from pathlib import Path

import torch

from spectrumsentry.catalog import Recording
from spectrumsentry.data import Span
from spectrumsentry.gate import Gate, frame_power

REC = Recording(Path("X_2G.bin"), "X", "2G", "", 0)
SPAN = Span(REC, 0, 10**6, 0)


def noise(b=2, n=8192, scale=0.01, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(b, n, generator=g), torch.randn(b, n, generator=g)) * scale


def test_frame_power_shape_and_value():
    x = torch.ones(1, 1024, dtype=torch.complex64)
    p = frame_power(x, nfft=256)
    assert p.shape == (1, 4) and torch.allclose(p, torch.ones(1, 4))


def test_gate_passes_a_burst_and_blocks_pure_noise():
    floor = float(frame_power(noise()).median())
    gate = Gate({"X_2G.bin": floor}, min_occupancy=0.3, margin_db=6.0)
    quiet = noise(b=1, seed=1)
    busy = quiet.clone()
    busy[:, 2048:6144] += noise(b=1, scale=0.5, seed=2)[:, :4096]  # half the window is strong signal
    items = [(SPAN, 0)]
    assert not gate.mask(quiet, items).item()
    assert gate.mask(busy, items).item()
    assert 0.4 < gate.occupancy(busy, items).item() < 0.6


def test_a_sparse_burst_below_the_occupancy_threshold_is_blocked():
    floor = float(frame_power(noise()).median())
    gate = Gate({"X_2G.bin": floor}, min_occupancy=0.3)
    x = noise(b=1, seed=3)
    x[:, 0:512] += noise(b=1, scale=0.5, seed=4)[:, :512]  # 2 of 32 frames
    assert not gate.mask(x, [(SPAN, 0)]).item()
