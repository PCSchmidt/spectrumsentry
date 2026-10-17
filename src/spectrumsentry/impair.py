"""Channel and receiver impairments applied to complex IQ windows (B, N).

These turn clean chamber recordings into stand-ins for other conditions. They are
simulations, not captures: see docs/design.md for what they do and do not represent.
All functions take a torch.Generator where randomness is involved and return a new tensor.
"""

from __future__ import annotations

import math

import torch


def awgn(x: torch.Tensor, snr_db: float, gen: torch.Generator | None = None) -> torch.Tensor:
    """Add complex white noise so that (mean power of x) / (noise power) equals ``snr_db`` per window.

    Signal power is the mean over the whole window, including gaps between hopping bursts,
    and the recording's own noise floor counts as signal: the real SNR is lower than stated.
    """
    p = x.abs().square().mean(dim=-1, keepdim=True)
    sigma = torch.sqrt(p / 10 ** (snr_db / 10) / 2)
    noise = torch.complex(
        torch.randn(x.shape, generator=gen, device=x.device),
        torch.randn(x.shape, generator=gen, device=x.device),
    )
    return x + sigma * noise


def cfo(x: torch.Tensor, offset_hz: torch.Tensor | float, fs: float) -> torch.Tensor:
    """Carrier frequency offset: multiply by exp(j 2 pi f n / fs); ``offset_hz`` may be per-window (B,)."""
    n = torch.arange(x.shape[-1], device=x.device, dtype=torch.float32)
    f = torch.as_tensor(offset_hz, dtype=torch.float32, device=x.device).reshape(-1, 1)
    phase = 2 * math.pi * f * n / fs
    return x * torch.polar(torch.ones_like(phase), phase)


def iq_imbalance(x: torch.Tensor, gain_db: float, phase_deg: float) -> torch.Tensor:
    """Receiver I/Q gain mismatch and quadrature error: Q' = g (Q cos phi + I sin phi)."""
    g = 10 ** (gain_db / 20)
    phi = math.radians(phase_deg)
    i, q = x.real, x.imag
    return torch.complex(i, g * (q * math.cos(phi) + i * math.sin(phi)))


def multipath(x: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
    """Circular convolution with complex channel taps (B, K) or (K,), via FFT. Taps are normalised to unit energy."""
    taps = taps / taps.abs().square().sum(dim=-1, keepdim=True).sqrt()
    h = torch.zeros_like(x) if taps.dim() == 2 else torch.zeros(x.shape[-1], dtype=x.dtype, device=x.device)
    h[..., : taps.shape[-1]] = taps
    return torch.fft.ifft(torch.fft.fft(x, dim=-1) * torch.fft.fft(h, dim=-1), dim=-1)


def random_multipath(batch: int, n_taps: int = 4, decay: float = 0.5, device="cpu", gen: torch.Generator | None = None) -> torch.Tensor:
    """Rayleigh taps with exponentially decaying power; the first tap is the direct path."""
    power = decay ** torch.arange(n_taps, dtype=torch.float32, device=device)
    re = torch.randn(batch, n_taps, generator=gen, device=device) * power.sqrt()
    im = torch.randn(batch, n_taps, generator=gen, device=device) * power.sqrt()
    return torch.complex(re, im)


def bandlimit(x: torch.Tensor, fs: float, bandwidth_hz: float) -> torch.Tensor:
    """Brick-wall filter: keep only +/- bandwidth/2 around the capture centre (a narrower receiver)."""
    spec = torch.fft.fft(x, dim=-1)
    freqs = torch.fft.fftfreq(x.shape[-1], d=1 / fs, device=x.device)
    return torch.fft.ifft(spec * (freqs.abs() <= bandwidth_hz / 2), dim=-1)


def quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Re-quantise I and Q to ``bits`` bits using the window's own peak as full scale (an idealised ADC with AGC)."""
    peak = torch.maximum(x.real.abs().amax(dim=-1, keepdim=True), x.imag.abs().amax(dim=-1, keepdim=True)).clamp_min(1e-12)
    levels = 2 ** (bits - 1) - 1

    def q(v):
        return torch.round(v / peak * levels) / levels * peak

    return torch.complex(q(x.real), q(x.imag))
