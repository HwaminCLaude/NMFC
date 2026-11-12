import torch

from src.nmfc import (
    geometric_logits,
    local_energies,
    mean_scale,
    median_sigma2,
    pairwise_sqdist,
    soft_neighbor_weights,
)


def make_batch(seed=0, n_per=8, m=4, gap=6.0):
    g = torch.Generator().manual_seed(seed)
    z0 = torch.randn(n_per, m, generator=g)
    z1 = torch.randn(n_per, m, generator=g) + gap
    z = torch.cat([z0, z1])
    y = torch.cat([torch.zeros(n_per, dtype=torch.long), torch.ones(n_per, dtype=torch.long)])
    return z, y


def test_pi_rows_sum_to_one_and_diag_zero():
    z, y = make_batch()
    d2 = pairwise_sqdist(z, z)
    sigma2 = median_sigma2(d2)
    classes = torch.unique(y)
    pi_pos, pi_neg = soft_neighbor_weights(d2, y, classes, sigma2)
    assert torch.allclose(pi_pos.sum(-1), torch.ones_like(pi_pos.sum(-1)), atol=1e-5)
    assert torch.allclose(pi_neg.sum(-1), torch.ones_like(pi_neg.sum(-1)), atol=1e-5)
    n = len(z)
    eye = torch.eye(n, dtype=torch.bool)
    assert pi_pos[:, eye].abs().max() == 0
    assert pi_neg[:, eye].abs().max() == 0


def test_energies_positive_and_monotone_with_distance():
    """프로브가 클래스에서 멀어질수록 E+가 단조 증가해야 한다.

    문자 그대로의 a/(Σa+ε) 구현은 원거리에서 언더플로로 E+→0이 되어
    이 테스트가 잡아낸다 (마스크드 softmax는 soft-min 거리를 유지).
    """
    g = torch.Generator().manual_seed(1)
    bank = torch.randn(32, 2, generator=g)  # 클래스 0 군집
    y = torch.zeros(32, dtype=torch.long)
    classes = torch.tensor([0])
    sigma2 = median_sigma2(pairwise_sqdist(bank, bank))
    prev = -1.0
    for dist in [1.0, 5.0, 20.0, 100.0, 1000.0]:
        probe = torch.tensor([[dist, 0.0]])
        d2 = pairwise_sqdist(probe, bank)
        pi_pos, _ = soft_neighbor_weights(d2, y, classes, sigma2, exclude_self=False)
        e_pos, _ = local_energies(d2, pi_pos, torch.zeros_like(pi_pos))
        val = e_pos.item()
        assert val > 0
        assert val > prev, f"dist={dist}에서 E+ 단조성 위반: {val} <= {prev}"
        prev = val


def test_logits_prefer_own_cluster():
    z, y = make_batch()
    d2 = pairwise_sqdist(z, z)
    sigma2 = median_sigma2(d2)
    classes, y_local = torch.unique(y, return_inverse=True)
    pi_pos, pi_neg = soft_neighbor_weights(d2, y, classes, sigma2)
    e_pos, e_neg = local_energies(d2, pi_pos, pi_neg)
    mu_pos, mu_neg = mean_scale(e_pos, e_neg, y_local)
    logits = geometric_logits(e_pos, e_neg, mu_pos, mu_neg)
    assert (logits.argmax(1) == y_local).float().mean() == 1.0
