import math
from pathlib import Path

import numpy as np
import pytest
import torch

from spectrumsentry import impair
from spectrumsentry.catalog import parse
from spectrumsentry.iq import load_int16, windows
from spectrumsentry.model import SpecCnn
from spectrumsentry.spec import spectrogram


def tone(n=4096, f=0.1, b=2):
    t = torch.arange(n, dtype=torch.float32)
    return torch.polar(torch.ones(b, n), 2 * math.pi * f * t).to(torch.complex64)


def test_catalog_parses_names():
    r = parse(Path("DJI_inspire_2_5G_2of2.bin"))
    assert (r.drone, r.band, r.part, r.fs, r.n_samples) == ("DJI_inspire_2", "5G", 2, 200e6, 100_000_000)
    m = parse(Path("Parrot_mambo_video_2G.bin"))
    assert (m.drone, m.link, m.part) == ("Parrot_mambo", "video", 0)
    assert parse(Path("Yuneec_typhoon_h_2G_1of2.bin")).drone == "Yuneec_typhoon_h"
    with pytest.raises(ValueError):
        parse(Path("load_bin.py"))


def test_windows_reads_interleaved_int16(tmp_path):
    data = np.array([[100, -200], [300, 400], [-500, 600], [700, -800]], dtype="<i2")
    p = tmp_path / "x.bin"
    data.tofile(p)
    w = windows(load_int16(p), np.array([1]), 2)
    assert w.shape == (1, 2)
    assert w[0, 0] == pytest.approx(complex(300, 400) / 32768)
    assert w[0, 1] == pytest.approx(complex(-500, 600) / 32768)


def test_spectrogram_shape_and_tone_location():
    s = spectrogram(tone(f=0.25), nfft=64, out=16)
    assert s.shape == (2, 1, 16, 16)
    assert abs(s.mean().item()) < 1e-5  # standardised
    peak_bin = s[0, 0].mean(dim=0).argmax().item()
    assert peak_bin in (11, 12)  # +0.25 cycles/sample is 3/4 of the way across a DC-centred axis


def test_awgn_hits_target_snr():
    x = tone(n=1 << 16, b=4)
    y = impair.awgn(x, 10.0, torch.Generator().manual_seed(0))
    noise_p = (y - x).abs().square().mean()
    assert 10 * math.log10(1.0 / noise_p.item()) == pytest.approx(10.0, abs=0.3)


def test_cfo_shifts_the_tone():
    x = tone(f=0.0, b=1)
    y = impair.cfo(x, 0.1 * 1000.0, fs=1000.0)
    f = torch.fft.fftfreq(x.shape[-1])[torch.fft.fft(y, dim=-1).abs().argmax(dim=-1)]
    assert f.item() == pytest.approx(0.1, abs=1e-3)


def test_iq_imbalance_identity_and_mismatch():
    x = tone()
    assert torch.allclose(impair.iq_imbalance(x, 0.0, 0.0), x, atol=1e-6)
    y = impair.iq_imbalance(x, 1.0, 5.0)
    assert not torch.allclose(y, x)


def test_multipath_with_single_tap_is_identity():
    x = tone()
    y = impair.multipath(x, torch.tensor([1.0 + 0j]))
    assert torch.allclose(y, x, atol=1e-5)


def test_bandlimit_removes_out_of_band_tone():
    x = tone(f=0.4, b=1)
    y = impair.bandlimit(x, fs=1.0, bandwidth_hz=0.2)
    assert y.abs().mean() < 0.01
    inside = impair.bandlimit(tone(f=0.05, b=1), fs=1.0, bandwidth_hz=0.2)
    assert inside.abs().mean() > 0.9


def test_quantize_levels():
    x = torch.complex(torch.linspace(-1, 1, 1000)[None], torch.zeros(1, 1000))
    y = impair.quantize(x, 3)
    assert len(torch.unique(y.real)) <= 7
    assert (y - x).abs().max() <= 1 / 3 / 2 + 1e-6


def test_model_forward_shape():
    assert SpecCnn(10)(torch.randn(3, 1, 64, 64)).shape == (3, 10)
