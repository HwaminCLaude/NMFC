import pytest
import torch

from src.nmfc.acs import ACSController, fisher_j, neg_j_loss
from src.nmfc.apt import NONLINEAR
from src.nmfc.losses import linear_mfa_loss


def two_clusters(gap, seed=0, n_per=32, m=8):
    g = torch.Generator().manual_seed(seed)
    z0 = torch.randn(n_per, m, generator=g)
    z1 = torch.randn(n_per, m, generator=g) + gap
    z = torch.cat([z0, z1])
    y = torch.cat([torch.zeros(n_per, dtype=torch.long), torch.ones(n_per, dtype=torch.long)])
    return z, y


def test_fisher_j_matches_reciprocal_lmf():
    """fisher_j ≈ 1/linear_mfa_loss — 역수형과 −J형이 같은 J를 가리킨다."""
    z, y = two_clusters(gap=3.0)
    j = fisher_j(z, y)
    assert torch.allclose(j, 1.0 / linear_mfa_loss(z, y), atol=1e-3, rtol=1e-3)
    assert torch.allclose(neg_j_loss(z, y), -j)


def test_neg_j_gradient_separates_clusters():
    z, y = two_clusters(gap=1.5)
    z = z.clone().requires_grad_(True)
    neg_j_loss(z, y).backward()
    z_after = (z - 0.05 * z.grad).detach()
    assert fisher_j(z_after, y) > fisher_j(z.detach(), y)


# ---- v4: 단위 구면 -J(sphere)와 trace-ratio 절제 ----


def test_fisher_j_sphere_scale_invariant():
    """v4: 단위 구면 J는 임베딩 균일 스케일에 불변 (ẑ=z/‖z‖이 z→az에 불변)."""
    z, y = two_clusters(gap=3.0)
    j1 = fisher_j(z, y, sphere=True)
    j2 = fisher_j(3.7 * z, y, sphere=True)
    assert torch.allclose(j1, j2, atol=1e-4, rtol=1e-4)


def test_sphere_negj_no_radial_gradient():
    """v4 핵심: sphere -J 손실은 ẑ에만 의존 → 각 z_i의 반경(팽창) 방향 grad가 정확히 0.
    (임베딩을 부풀려 이득을 취할 자유도가 척도 불변성으로 닫힘.)"""
    z, y = two_clusters(gap=1.5)
    zs = z.clone().requires_grad_(True)
    neg_j_loss(zs, y, sphere=True).backward()
    zhat = z / z.norm(dim=1, keepdim=True)
    radial = (zs.grad * zhat).sum(dim=1).abs().max()
    assert radial < 1e-4


def test_sphere_j_bounded_under_mean_spreading():
    """원공간 -J는 클래스 평균 벌림(S_B∝gap²)으로 무한히 커지지만(팽창 유인), sphere -J는
    단위 구면상 유계라 훨씬 덜 자란다(v4 팽창 차단)."""
    z_near, y = two_clusters(gap=3.0)
    z_far, _ = two_clusters(gap=60.0)
    raw_ratio = (fisher_j(z_far, y) / fisher_j(z_near, y)).item()
    sph_ratio = (fisher_j(z_far, y, sphere=True) / fisher_j(z_near, y, sphere=True)).item()
    assert raw_ratio > 5.0
    assert sph_ratio < raw_ratio


def test_trace_ratio_bounded_and_excludes_sphere():
    """v4 절제: -J/(J+κ)는 (-1, 0]에 유계(J≥0, κ>0). sphere와 kappa 동시 지정은 금지."""
    z, y = two_clusters(gap=8.0)
    j = fisher_j(z, y).item()
    loss = neg_j_loss(z, y, kappa=max(j, 1.0)).item()
    assert -1.0 < loss <= 0.0
    with pytest.raises(AssertionError):
        neg_j_loss(z, y, sphere=True, kappa=1.0)


def test_acs_triggers_on_plateau():
    acs = ACSController(window=2, patience=2, delta=1e-3, t_min=4, t_max=30, ramp=3)
    seq = [float(t) for t in range(1, 6)] + [10.0] * 10  # 상승 후 plateau
    fired = None
    for j in seq:
        if acs.update(j) == NONLINEAR and fired is None:
            fired = len(acs.js)
    assert fired is not None and acs.transition_reason == "converged"


def test_acs_max_warmup_forces_transition():
    acs = ACSController(window=2, patience=2, delta=1e-4, t_min=4, t_max=8, ramp=3)
    for t in range(1, 20):
        acs.update(2.0 * t)  # 계속 상승 → 정체 감지 불가
    assert acs.phase == NONLINEAR
    assert acs.transition_epoch == 8 and acs.transition_reason == "max_warmup"


def test_acs_eps_phi_guard_when_j_zero():
    """J=0(붕괴 임베딩) 연속 시 분모 max(0, ε_Φ)=ε_Φ로 0-나눗셈 없이 안정 발동."""
    acs = ACSController(window=1, patience=1, delta=1e-3, t_min=2, t_max=30, ramp=2)
    for _ in range(3):
        acs.update(0.0)
    assert acs.phase == NONLINEAR
    assert all(p == 0.0 for p in acs.phis)


def test_acs_alpha_ramp():
    acs = ACSController(window=2, patience=2, delta=1e-3, t_min=4, t_max=30, ramp=4)
    acs._fire(5, "converged")
    assert acs.alpha(4) == 0.0   # 전환 전
    assert acs.alpha(5) == 0.0   # 전환 에폭
    assert acs.alpha(6) == 0.25  # (6−5)/4
    assert acs.alpha(9) == 1.0   # (9−5)/4
    assert acs.alpha(12) == 1.0  # 클램프


def test_acs_asserts_window_patience_le_tmin():
    with pytest.raises(AssertionError):
        ACSController(window=3, patience=3, t_min=5)  # 6 > 5
