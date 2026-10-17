import torch

from spectrumsentry.openset import Mahalanobis, auroc, energy, msp, unknown_accepted_at_tpr


def test_auroc_extremes_and_ties():
    assert auroc(torch.tensor([3.0, 4.0]), torch.tensor([1.0, 2.0])) == 1.0
    assert auroc(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])) == 0.0
    assert auroc(torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0])) == 0.5


def test_auroc_matches_pairwise_definition():
    g = torch.Generator().manual_seed(0)
    k, u = torch.randn(40, generator=g) + 1, torch.randn(50, generator=g)
    pairs = ((k[:, None] > u[None]).double() + 0.5 * (k[:, None] == u[None]).double()).mean().item()
    assert abs(auroc(k, u) - pairs) < 1e-9


def test_msp_and_energy_are_higher_for_confident_logits():
    confident, flat = torch.tensor([[10.0, 0.0, 0.0]]), torch.tensor([[1.0, 1.0, 1.0]])
    assert msp(confident) > msp(flat) and energy(confident) > energy(flat)


def test_mahalanobis_scores_known_clusters_above_far_points():
    g = torch.Generator().manual_seed(1)
    feats = torch.cat([torch.randn(200, 4, generator=g) * 0.3 + 5, torch.randn(200, 4, generator=g) * 0.3 - 5])
    labels = torch.tensor([0] * 200 + [1] * 200)
    m = Mahalanobis(feats, labels, 2)
    inlier = torch.randn(50, 4, generator=g) * 0.3 + 5
    outlier = torch.zeros(50, 4)  # between the clusters
    assert auroc(m.score(inlier), m.score(outlier)) > 0.99


def test_threshold_keeps_the_requested_share_of_known():
    known = torch.linspace(0, 1, 1001)
    unknown = torch.tensor([0.0, 0.02, 0.5, 0.9])
    thr, accepted = unknown_accepted_at_tpr(known, unknown, 0.95)
    assert abs(thr - 0.05) < 1e-3 and accepted == 0.5
