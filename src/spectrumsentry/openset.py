"""Open-set scores: how much does a window look like a drone the model was trained on?

Higher score = more likely a known class. Three standard post-hoc scores on a trained classifier:

* ``msp``     maximum softmax probability (Hendrycks and Gimpel, 2017)
* ``energy``  logsumexp of the logits (Liu et al., 2020)
* ``maha``    negative Mahalanobis distance to the nearest class mean in the penultimate-feature space,
              with one covariance shared by all classes (Lee et al., 2018)
"""

from __future__ import annotations

import numpy as np
import torch


def msp(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1).amax(dim=1)


def energy(logits: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(logits, dim=1)


class Mahalanobis:
    """Class means and a shared covariance fitted on training embeddings."""

    def __init__(self, feats: torch.Tensor, labels: torch.Tensor, n_classes: int, ridge: float = 1e-3):
        feats = feats.double()
        self.means = torch.stack([feats[labels == c].mean(0) for c in range(n_classes)])
        centred = feats - self.means[labels]
        cov = centred.T @ centred / len(feats)
        self.precision = torch.linalg.inv(cov + ridge * torch.trace(cov) / cov.shape[0] * torch.eye(cov.shape[0], dtype=torch.float64))

    def score(self, feats: torch.Tensor) -> torch.Tensor:
        diff = feats.double()[:, None, :] - self.means[None]  # (B, C, D)
        d = torch.einsum("bcd,de,bce->bc", diff, self.precision, diff)
        return -d.min(dim=1).values.float()


def auroc(known_scores: torch.Tensor, unknown_scores: torch.Tensor) -> float:
    """Probability that a random known window scores higher than a random unknown one (ties count half)."""
    k, u = known_scores.double().numpy(), unknown_scores.double().numpy()
    order = np.argsort(np.concatenate([k, u]), kind="mergesort")
    ranks = np.empty(len(order))
    all_scores = np.concatenate([k, u])[order]
    i = 0
    while i < len(all_scores):  # average ranks over ties
        j = i
        while j + 1 < len(all_scores) and all_scores[j + 1] == all_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[: len(k)].sum() - len(k) * (len(k) + 1) / 2) / (len(k) * len(u)))


def unknown_accepted_at_tpr(known_scores: torch.Tensor, unknown_scores: torch.Tensor, tpr: float = 0.95) -> tuple[float, float]:
    """(threshold, share of unknown windows accepted) when the threshold keeps ``tpr`` of known windows."""
    thr = torch.quantile(known_scores.double(), 1 - tpr).item()
    return thr, float((unknown_scores.double() >= thr).double().mean().item())
