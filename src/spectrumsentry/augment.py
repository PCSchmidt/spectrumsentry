"""Composable simulated shift: a spec of impairments, and a random augmenter that draws specs.

A spec is a dict with any of these keys (missing key = impairment off):

    mp   (decay, spacing)  4-tap Rayleigh multipath; spacing in samples (8.3 ns each at 120 MS/s)
    cfo  max_hz            carrier frequency offset, uniform in +/- max_hz per window
    snr  db                added white noise, window mean power / noise power
    bw   hz                brick-wall receiver bandwidth around the capture centre
    iq   (gain_db, deg)    I/Q gain and phase imbalance
    bits adc_bits          re-quantise to this many bits (idealised AGC)

Applied in the physical order: channel (multipath), oscillator (cfo), noise, receiver
filter (bw), mixer (iq), ADC (bits). All simulations, see docs/design.md.
"""

from __future__ import annotations

import torch

from . import impair

Spec = dict


def apply(x: torch.Tensor, fs: torch.Tensor, spec: Spec, gen: torch.Generator | None = None) -> torch.Tensor:
    if "mp" in spec:
        decay, spacing = spec["mp"]
        x = impair.multipath(x, impair.random_multipath(x.shape[0], 4, decay, spacing, x.device, gen))
    if "cfo" in spec:
        offs = (torch.rand(x.shape[0], generator=gen, device=x.device) * 2 - 1) * spec["cfo"]
        x = impair.cfo(x, offs, fs)
    if "snr" in spec:
        x = impair.awgn(x, spec["snr"], gen)
    if "bw" in spec:
        x = impair.bandlimit(x, fs, spec["bw"])
    if "iq" in spec:
        x = impair.iq_imbalance(x, *spec["iq"])
    if "bits" in spec:
        x = impair.quantize(x, spec["bits"])
    return x


def fixed(spec: Spec, seed: int = 0):
    """A transform applying ``spec`` with its own repeatable random stream (created lazily on the batch's device)."""
    state: dict = {}

    def transform(x: torch.Tensor, fs: torch.Tensor) -> torch.Tensor:
        if "gen" not in state:
            state["gen"] = torch.Generator(device=x.device).manual_seed(seed)
        return apply(x, fs, spec, state["gen"])

    return transform


class RandomAugment:
    """Draws a random spec per batch. ``receiver=False`` leaves out bandwidth limiting and quantisation
    (to test generalisation to receiver impairments the model never saw)."""

    def __init__(self, receiver: bool = True, seed: int = 0):
        self.receiver = receiver
        self.rng = torch.Generator().manual_seed(seed)

    def _u(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * torch.rand(1, generator=self.rng).item()

    def _p(self, p: float) -> bool:
        return torch.rand(1, generator=self.rng).item() < p

    def spec(self) -> Spec:
        s: Spec = {}
        if self._p(0.5):
            s["mp"] = (self._u(0.2, 0.8), int(self._u(1, 7)))
        if self._p(0.5):
            s["cfo"] = self._u(0.0, 1e6)
        if self._p(0.8):
            s["snr"] = self._u(-5.0, 30.0)
        if self.receiver and self._p(0.5):
            s["bw"] = [56e6, 20e6, 10e6][int(self._u(0, 3))]
        if self._p(0.5):
            s["iq"] = (self._u(0.0, 2.0), self._u(0.0, 15.0))
        if self.receiver and self._p(0.5):
            s["bits"] = [12, 8, 6][int(self._u(0, 3))]
        return s

    def __call__(self, x: torch.Tensor, fs: torch.Tensor) -> torch.Tensor:
        return apply(x, fs, self.spec(), None)
