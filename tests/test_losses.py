import torch

from src.nmfc import fisher_ratio, linear_mfa_loss, nmfc_loss, scatter_matrices


def two_clusters(gap, seed=0, n_per=32, m=8):
    g = torch.Generator().manual_seed(seed)
    z0 = torch.randn(n_per, m, generator=g)
    z1 = torch.randn(n_per, m, generator=g) + gap
    z = torch.cat([z0, z1])
    y = torch.cat([torch.zeros(n_per, dtype=torch.long), torch.ones(n_per, dtype=torch.long)])
    return z, y


def test_scatter_traces_positive():
    z, y = two_clusters(gap=3.0)
    s_w, s_b = scatter_matrices(z, y)
    assert s_w.diagonal().sum() > 0
    assert s_b.diagonal().sum() > 0
    assert torch.allclose(s_w, s_w.T, atol=1e-5)


def test_fisher_ratio_increases_with_separation():
    z1, y = two_clusters(gap=1.0)
    z2, _ = two_clusters(gap=10.0)
    assert fisher_ratio(z2, y) > fisher_ratio(z1, y)


def test_lmf_loss_decreases_with_separation():
    """L_LMF 최소화 = Fisher 기준 최대화 — 잘 분리된 임베딩일수록 손실이 작아야 한다."""
    z1, y = two_clusters(gap=1.0)
    z2, _ = two_clusters(gap=10.0)
    assert linear_mfa_loss(z2, y) < linear_mfa_loss(z1, y)
    assert linear_mfa_loss(z1, y) > 0


def test_lmf_gradient_separates_clusters():
    """L_LMF의 경사 하강 한 걸음이 실제로 Fisher ratio를 개선해야 한다."""
    z, y = two_clusters(gap=1.5)
    z = z.clone().requires_grad_(True)
    loss = linear_mfa_loss(z, y)
    loss.backward()
    z_after = (z - 0.5 * z.grad).detach()
    assert fisher_ratio(z_after, y) > fisher_ratio(z.detach(), y)


def test_nmfc_loss_prefers_correct_logits():
    g_good = torch.tensor([[5.0, -5.0], [-5.0, 5.0]])
    g_bad = torch.tensor([[-5.0, 5.0], [5.0, -5.0]])
    y = torch.tensor([0, 1])
    assert nmfc_loss(g_good, y) < nmfc_loss(g_bad, y)
