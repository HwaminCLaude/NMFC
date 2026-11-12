import math

import torch

from src.nmfc.softmin import knn_neighbor_sets, softmin_energies


def test_softmin_two_point_exact():
    """k₁=1이면 soft-min = 최근접 제곱거리 그대로."""
    z = torch.tensor([[0.0, 0.0]])
    ref = torch.tensor([[3.0, 0.0], [5.0, 0.0]])  # 클래스 0 참조 2개, d²=9, 25
    d2 = torch.cdist(z, ref).pow(2)
    y_col = torch.tensor([0, 0])
    classes = torch.tensor([0])
    idx_pos, _ = knn_neighbor_sets(d2, y_col, classes, k1=1, k2=1)
    e_pos = softmin_energies(d2, idx_pos, torch.tensor(1.0))
    assert torch.allclose(e_pos, torch.tensor([[9.0]]), atol=1e-5)


def test_prop1_energy_bound():
    """명제 1: d_min² − σ²·ln k₁ ≤ E⁺ ≤ d_min²."""
    g = torch.Generator().manual_seed(1)
    ref = torch.randn(15, 4, generator=g)
    y_col = torch.arange(3).repeat_interleave(5)
    z = torch.randn(6, 4, generator=g)
    d2 = torch.cdist(z, ref).pow(2)
    classes = torch.tensor([0, 1, 2])
    sigma2 = torch.tensor(0.7)
    k1 = 3
    idx_pos, _ = knn_neighbor_sets(d2, y_col, classes, k1=k1, k2=3)
    e_pos = softmin_energies(d2, idx_pos, sigma2)  # (6, 3)
    for p, c in enumerate(classes):
        cols = (y_col == c).nonzero(as_tuple=True)[0]
        dmin2 = d2[:, cols].min(dim=1).values
        assert torch.all(e_pos[:, p] <= dmin2 + 1e-4)
        assert torch.all(e_pos[:, p] >= dmin2 - sigma2 * math.log(k1) - 1e-4)


def test_loo_excludes_self_and_tag():
    """배치 내 LOO: 자기 열(거리 0)을 제외해야 E⁺가 0으로 붕괴하지 않는다."""
    z = torch.tensor([[0.0, 0.0], [0.1, 0.0], [3.0, 0.0], [3.1, 0.0]])
    y = torch.tensor([0, 0, 1, 1])
    d2 = torch.cdist(z, z).pow(2)
    classes = torch.tensor([0, 1])
    exclude = torch.arange(4)  # 각 쿼리가 자기 행 제외
    idx_pos, _ = knn_neighbor_sets(d2, y, classes, k1=1, k2=1, exclude_cols=exclude)
    e_pos = softmin_energies(d2, idx_pos, torch.tensor(1.0))
    assert idx_pos[0, 0, 0].item() != 0  # 쿼리0이 자기 열을 고르지 않음
    assert e_pos[0, 0] > 1e-3  # 자기(0) 대신 이웃1(d²=0.01)

    # LOO 없이는 자기(d²=0) 선택 → E⁺≈0
    idx_no_loo, _ = knn_neighbor_sets(d2, y, classes, k1=1, k2=1)
    e_no_loo = softmin_energies(d2, idx_no_loo, torch.tensor(1.0))
    assert e_no_loo[0, 0].abs() < 1e-5


def test_grad_flows_to_query_only():
    """그래디언트는 gather된 거리(쿼리)로만 흐르고, 이웃 선택 인덱스엔 없다(long)."""
    g = torch.Generator().manual_seed(2)
    ref = torch.randn(9, 3, generator=g)
    y_col = torch.arange(3).repeat_interleave(3)
    z = torch.randn(4, 3, generator=g, requires_grad=True)
    d2 = torch.cdist(z, ref).pow(2)
    classes = torch.tensor([0, 1, 2])
    idx_pos, idx_neg = knn_neighbor_sets(d2, y_col, classes, k1=2, k2=2)
    assert idx_pos.dtype == torch.long and not idx_pos.requires_grad
    e_pos = softmin_energies(d2, idx_pos, torch.tensor(0.5))
    e_pos.sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_softmin_gradient_matches_closed_form():
    """경사식: ∂E⁺/∂z_i = 2·Σ_j w⁺_ij (z_i − z_j), w⁺=soft-min 배정(집합 내 합=1)."""
    g = torch.Generator().manual_seed(3)
    ref = torch.randn(6, 3, generator=g)
    y_col = torch.tensor([0, 0, 0, 1, 1, 1])
    z = torch.randn(1, 3, generator=g, requires_grad=True)
    d2 = torch.cdist(z, ref).pow(2)
    sigma2 = torch.tensor(0.8)
    idx_pos, _ = knn_neighbor_sets(d2, y_col, torch.tensor([0]), k1=3, k2=1)
    e_pos = softmin_energies(d2, idx_pos, sigma2)
    (grad,) = torch.autograd.grad(e_pos[0, 0], z)
    with torch.no_grad():
        nbr = idx_pos[0, 0]
        w = torch.softmax(-d2.detach()[0, nbr] / sigma2, dim=0)
        analytic = 2 * (w[:, None] * (z.detach() - ref[nbr])).sum(0, keepdim=True)
    assert torch.allclose(grad, analytic, atol=1e-4)
