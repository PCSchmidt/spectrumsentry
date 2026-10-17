from pathlib import Path

import numpy as np
import torch

from spectrumsentry import dronerf as D


def make_dir(tmp_path, codes=("00000", "10100"), n=10):
    for code in codes:
        for half in "LH":
            for i in range(n):
                np.save(tmp_path / f"{code}{half}_{i}.npy", (np.random.default_rng(i).normal(0, 50, 20_000)).astype(np.int16))
    return tmp_path


def test_catalog_pairs_halves_and_maps_codes(tmp_path):
    segs = D.catalog(make_dir(tmp_path))
    assert len(segs) == 20
    assert {s.drone for s in segs} == {"background", "ar"}
    assert all(s.low.name.replace("L", "H") == s.high.name for s in segs)


def test_split_holds_out_whole_segments_in_order(tmp_path):
    segs = D.catalog(make_dir(tmp_path))
    parts = D.split(segs, "drone")
    keys = {k: {(s.bui, s.idx) for s in v} for k, v in parts.items()}
    assert not (keys["train"] & keys["val"]) and not (keys["train"] & keys["test"]) and not (keys["val"] & keys["test"])
    assert len(keys["train"]) == 12 and len(keys["val"]) == 2 and len(keys["test"]) == 6
    assert max(i for _, i in keys["train"]) < min(i for _, i in keys["test"])  # earlier segments train, later test


def test_phantom_and_bebop_codes():
    mk = lambda b: D.Seg(b, 0, Path("a"), Path("b"))  # noqa: E731
    assert [mk(b).drone for b in ("10000", "10011", "10100", "11000", "00000")] == ["bebop", "bebop", "ar", "phantom", "background"]


def test_spectrogram_shape_and_standardised():
    x = torch.randn(3, 2, 4096)
    s = D.spectrogram(x)
    assert s.shape == (3, 2, 64, 64)
    assert abs(s.mean().item()) < 1e-4


def test_awgn_real_hits_target_snr():
    x = torch.randn(4, 2, 1 << 16)
    y = D.awgn_real(x, 10.0, torch.Generator().manual_seed(0))
    snr = 10 * torch.log10(x.square().mean() / (y - x).square().mean()).item()
    assert abs(snr - 10.0) < 0.3


def test_bandlimit_real_removes_high_tone():
    t = torch.arange(8192, dtype=torch.float32)
    x = torch.sin(2 * torch.pi * 0.4 * t)[None, None]
    assert D.bandlimit_real(x, 0.5).abs().mean() < 0.01
    assert D.bandlimit_real(torch.sin(2 * torch.pi * 0.05 * t)[None, None], 0.5).abs().mean() > 0.5


def test_quantize_real_levels():
    x = torch.linspace(-1, 1, 1000)[None, None]
    assert len(torch.unique(D.quantize_real(x, 3))) <= 7


def test_interleaved_split_has_no_shared_segments_and_mixes_time(tmp_path):
    segs = D.catalog(make_dir(tmp_path, n=20))
    parts = D.split(segs, "drone", "interleaved")
    keys = {k: {(s.bui, s.idx) for s in v} for k, v in parts.items()}
    assert not (keys["train"] & keys["test"]) and not (keys["train"] & keys["val"]) and not (keys["val"] & keys["test"])
    assert sum(len(v) for v in keys.values()) == len(segs)
    # test segments occur both early and late, unlike the ordered split
    test_idx = sorted(i for _, i in keys["test"])
    assert test_idx[0] < 5 and test_idx[-1] > 14
