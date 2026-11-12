import numpy as np
import torch

from src.models.banks import kcenter_greedy


def test_kcenter_deterministic():
    z = torch.randn(50, 4, generator=torch.Generator().manual_seed(0))
    s1 = kcenter_greedy(z, 10, seed=7)
    s2 = kcenter_greedy(z, 10, seed=7)
    assert torch.equal(s1, s2)


def test_kcenter_size_clamped():
    z = torch.randn(5, 3)
    assert len(kcenter_greedy(z, 10)) == 5  # min(n_c, len(z))


def test_kcenter_better_coverage_than_random():
    """planted 군집에서 k-center가 랜덤보다 상호 최소거리(커버리지)가 크다."""
    g = torch.Generator().manual_seed(1)
    centers = torch.tensor([[0.0, 0], [10, 0], [0, 10], [10, 10], [5, 5]])
    z = torch.cat([c + 0.1 * torch.randn(20, 2, generator=g) for c in centers])
    picked = z[kcenter_greedy(z, 5, seed=0)]
    kc_min = torch.cdist(picked, picked).fill_diagonal_(float("inf")).min().item()

    rng = np.random.default_rng(0)
    rand_mins = []
    for _ in range(20):
        pr = z[torch.tensor(rng.choice(len(z), 5, replace=False))]
        rand_mins.append(torch.cdist(pr, pr).fill_diagonal_(float("inf")).min().item())
    assert kc_min > float(np.mean(rand_mins))
