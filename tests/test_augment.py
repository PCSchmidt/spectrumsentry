import math

import torch

from spectrumsentry import augment


def tone(b=3, n=4096, f=0.1):
    t = torch.arange(n, dtype=torch.float32)
    return torch.polar(torch.ones(b, n), 2 * math.pi * f * t).to(torch.complex64)


FS = torch.full((3,), 120e6)


def test_empty_spec_is_identity():
    x = tone()
    assert torch.equal(augment.apply(x, FS, {}), x)


def test_every_impairment_changes_the_signal_and_keeps_shape_and_dtype():
    x = tone()
    for spec in ({"snr": 5.0}, {"cfo": 1e6}, {"iq": (2.0, 10.0)}, {"mp": (0.5, 3)}, {"bw": 10e6}, {"bits": 4}):
        y = augment.apply(x, FS, spec, torch.Generator().manual_seed(0))
        assert y.shape == x.shape and y.dtype == x.dtype and not torch.allclose(y, x), spec


def test_fixed_transform_is_repeatable():
    x = tone()
    a = augment.fixed({"snr": 0.0, "cfo": 1e5}, seed=1)(x, FS)
    b = augment.fixed({"snr": 0.0, "cfo": 1e5}, seed=1)(x, FS)
    assert torch.equal(a, b)


def test_per_window_sample_rate_is_respected_by_bandlimit():
    # a 20 MHz tone survives a 56 MHz filter, a 40 MHz tone does not, whatever the sample rate
    fs = torch.tensor([120e6, 200e6])
    n = 8192
    t = torch.arange(n, dtype=torch.float32)
    x = torch.stack([torch.polar(torch.ones(n), 2 * math.pi * 20e6 / fs[0] * t), torch.polar(torch.ones(n), 2 * math.pi * 40e6 / fs[1] * t)]).to(torch.complex64)
    y = augment.apply(x, fs, {"bw": 56e6})
    assert y[0].abs().mean() > 0.9 and y[1].abs().mean() < 0.05


def test_random_augment_without_receiver_never_filters_or_quantises():
    aug = augment.RandomAugment(receiver=False, seed=0)
    assert all("bw" not in s and "bits" not in s for s in (aug.spec() for _ in range(200)))
    full = augment.RandomAugment(receiver=True, seed=0)
    specs = [full.spec() for _ in range(200)]
    assert any("bw" in s for s in specs) and any("bits" in s for s in specs)
