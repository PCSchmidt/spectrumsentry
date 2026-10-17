"""Training and evaluation loops for the spectrogram CNN."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import nn

from .data import Span, Store, eval_windows, load_batch, sample_from_candidates, sample_train
from .model import SpecCnn
from .spec import spectrogram

Transform = Callable[[torch.Tensor], torch.Tensor]  # complex IQ batch -> complex IQ batch


def train(
    spans: list[Span],
    n_classes: int,
    store: Store,
    device: str,
    steps: int = 600,
    batch: int = 48,
    seed: int = 0,
    transform: Transform | None = None,
    lr: float = 2e-3,
    candidates: dict[Span, np.ndarray] | None = None,
) -> SpecCnn:
    """Train from scratch. ``transform`` (e.g. random impairments) is applied to each training batch.

    With ``candidates`` (span -> start offsets that passed the energy detector), training windows are
    drawn only from those, still balanced across classes.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SpecCnn(n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(steps):
        items = sample_train(spans, batch, rng) if candidates is None else sample_from_candidates(spans, candidates, batch, rng)
        x, y = load_batch(store, items, device)
        if transform is not None:
            x = transform(x)
        loss = loss_fn(model(spectrogram(x)), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return model


@torch.no_grad()
def predict(model: SpecCnn, items, store: Store, device: str, transform: Transform | None = None, chunk: int = 48):
    """Logits (n, classes) and labels (n,) for fixed evaluation items."""
    model.eval()
    logits, labels = [], []
    for i in range(0, len(items), chunk):
        x, y = load_batch(store, items[i : i + chunk], device)
        if transform is not None:
            x = transform(x)
        logits.append(model(spectrogram(x)).cpu())
        labels.append(y.cpu())
    return torch.cat(logits), torch.cat(labels)


def accuracy(model: SpecCnn, spans: list[Span], store: Store, device: str, per_span: int = 60, transform: Transform | None = None) -> float:
    logits, labels = predict(model, eval_windows(spans, per_span), store, device, transform)
    return (logits.argmax(1) == labels).float().mean().item()
