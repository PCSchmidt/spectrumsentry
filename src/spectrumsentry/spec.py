"""Complex IQ windows to normalised log-power spectrogram images."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def spectrogram(x: torch.Tensor, nfft: int = 256, out: int = 64) -> torch.Tensor:
    """(B, N) complex IQ to (B, 1, out, out): log10 power, DC centred, pooled, standardised per example.

    Frames are non-overlapping ``nfft``-sample Hann-windowed FFTs. Per-example standardisation
    removes absolute level, so a receiver with a different gain does not change the picture.
    """
    b, n = x.shape
    frames = x[:, : n // nfft * nfft].reshape(b, -1, nfft) * torch.hann_window(nfft, device=x.device)
    power = torch.fft.fftshift(torch.fft.fft(frames, dim=-1), dim=-1).abs().square()  # (B, T, F)
    img = torch.log10(power + 1e-12).unsqueeze(1)
    img = F.adaptive_avg_pool2d(img, out)
    mean = img.mean(dim=(1, 2, 3), keepdim=True)
    std = img.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    return (img - mean) / std
