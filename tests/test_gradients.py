import torch

from src.nmfc import (
    geometric_logits,
    local_energies,
    mean_scale,
    median_sigma2,
    nmfc_loss,
    pairwise_sqdist,
    soft_neighbor_weights,
)


def two_clusters(gap=4.0, seed=0, n_per=16):
    g = torch.Generator().manual_seed(seed)
    c0 = torch.randn(n_per, 2, generator=g) * 0.3
    c1 = torch.randn(n_per, 2, generator=g) * 0.3 + torch.tensor([gap, 0.0])
    z = torch.cat([c0, c1])
    y = torch.cat([torch.zeros(n_per, dtype=torch.long), torch.ones(n_per, dtype=torch.long)])
    return z, y


def nmfc_logits(z, y, detach_pi=False):
    d2 = pairwise_sqdist(z, z)
    sigma2 = median_sigma2(d2)
    classes, y_local = torch.unique(y, return_inverse=True)
    pi_pos, pi_neg = soft_neighbor_weights(d2, y, classes, sigma2)
    if detach_pi:
        pi_pos, pi_neg = pi_pos.detach(), pi_neg.detach()
    e_pos, e_neg = local_energies(d2, pi_pos, pi_neg)
    mu_pos, mu_neg = mean_scale(e_pos, e_neg, y_local)
    return geometric_logits(e_pos, e_neg, mu_pos, mu_neg), y_local


def test_quasi_static_force_field_matches_closed_form():
    """경사식 검증 (준정적 π): 각 에너지 항의 경사가 정확히
    ∂E±_k/∂z_i = 2(z_i − Σ_j π±_ij z_j) 여야 한다.
    → -E⁺항은 π⁺ 가중 이웃 평균으로 '당기고', +λE⁻항은 π⁻ 평균에서 '밀어낸다'."""
    z, y = two_clusters()
    z = z.clone().requires_grad_(True)

    d2 = pairwise_sqdist(z, z)
    sigma2 = median_sigma2(d2)
    classes, _ = torch.unique(y, return_inverse=True)
    pi_pos, pi_neg = soft_neighbor_weights(d2, y, classes, sigma2)
    pi_pos_d, pi_neg_d = pi_pos.detach(), pi_neg.detach()
    e_pos, e_neg = local_energies(d2, pi_pos_d, pi_neg_d)

    i = 0  # 클래스 0 앵커
    zi = z[i].detach()
    nbr_pos = pi_pos_d[0, i] @ z.detach()  # π 가중 이웃 평균
    nbr_neg = pi_neg_d[0, i] @ z.detach()

    (grad_pos,) = torch.autograd.grad(e_pos[i, 0], z, retain_graph=True)
    (grad_neg,) = torch.autograd.grad(e_neg[i, 0], z)
    assert torch.allclose(grad_pos[i], 2 * (zi - nbr_pos), atol=1e-4), "E+ 경사가 닫힌 식과 다름"
    assert torch.allclose(grad_neg[i], 2 * (zi - nbr_neg), atol=1e-4), "E- 경사가 닫힌 식과 다름"
    # 부호 해석: 로짓의 -E+항 상승 방향은 자기 이웃 평균을 향하고(당김),
    # +λE-항 상승 방향은 상대 이웃 평균의 반대(밀침)
    assert torch.dot(-grad_pos[i], nbr_pos - zi) > 0, "당김(pull) 방향 위반"
    assert torch.dot(grad_neg[i], nbr_neg - zi) < 0, "밀침(push) 방향 위반"


def test_full_autograd_training_step_improves_loss():
    """full autograd(기본값) 경로의 행동 검증: 경사 하강 몇 걸음이
    NLL을 줄이고 정답률을 유지·개선해야 한다."""
    z, y = two_clusters(gap=1.2, seed=3)
    z = z.clone().requires_grad_(True)
    opt = torch.optim.SGD([z], lr=0.05)
    logits, y_local = nmfc_logits(z, y)
    loss0 = nmfc_loss(logits, y_local).item()
    for _ in range(20):
        opt.zero_grad()
        logits, y_local = nmfc_logits(z, y)
        loss = nmfc_loss(logits, y_local)
        loss.backward()
        opt.step()
    logits, y_local = nmfc_logits(z, y)
    assert nmfc_loss(logits, y_local).item() < loss0
    assert (logits.argmax(1) == y_local).float().mean() >= 0.9
