import math

import torch
import torch.nn.functional as F

from src.nmfc.softmin import hinge_temp_logits


def test_hinge_temp_logit_values():
    e_pos = torch.tensor([[2.0, 5.0]])
    e_neg = torch.tensor([[4.0, 1.0]])
    m = torch.tensor(3.0)
    tau = torch.tensor(2.0)
    g = hinge_temp_logits(e_pos, e_neg, 0.5, m, tau)
    # min(E⁻,m)=[3,1]; g=(-E⁺+0.5·min)/2 = ([-2+1.5, -5+0.5])/2 = [-0.25, -2.25]
    assert torch.allclose(g, torch.tensor([[-0.25, -2.25]]), atol=1e-6)


def test_hinge_cuts_gradient_above_margin():
    """E⁻>m이면 min(E⁻,m)=m(상수) → E⁻ 척력 grad 정확히 0 (지시함수 1[E⁻<m])."""
    m = torch.tensor(3.0)
    tau = torch.tensor(1.0)
    e_pos = torch.tensor([[1.0]])
    e_neg_hi = torch.tensor([[5.0]], requires_grad=True)
    hinge_temp_logits(e_pos, e_neg_hi, 0.5, m, tau).sum().backward()
    assert torch.allclose(e_neg_hi.grad, torch.zeros_like(e_neg_hi.grad))

    e_neg_lo = torch.tensor([[1.0]], requires_grad=True)
    hinge_temp_logits(e_pos, e_neg_lo, 0.5, m, tau).sum().backward()
    assert torch.allclose(e_neg_lo.grad, torch.tensor([[0.5]]))  # λ/τ


def test_tau_reparam_lower_bound_and_init():
    """τ = τ_min + softplus(ρ) > τ_min 항상; ρ₀ = log(exp(1−τ_min)−1) → τ₀ = 1.0."""
    tau_min = 0.01
    rho0 = math.log(math.exp(1.0 - tau_min) - 1.0)
    tau0 = tau_min + F.softplus(torch.tensor(rho0))
    assert abs(tau0.item() - 1.0) < 1e-5
    # 하한 보장의 본질: softplus(ρ) ≥ 0 ⟹ τ = τ_min + softplus(ρ) ≥ τ_min 항상
    for r in [-50.0, -5.0, 0.0, 5.0, 50.0]:
        assert F.softplus(torch.tensor(r)).item() >= 0.0
    for r in [-5.0, 0.0, 5.0]:
        assert (tau_min + F.softplus(torch.tensor(r))).item() > tau_min


def test_rho_receives_nll_grad():
    """ρ가 온도 τ를 통해 NLL 그래디언트를 받는다(정지 경사 걸지 않음)."""
    tau_min = 0.01
    rho = torch.zeros(1, requires_grad=True)
    tau = tau_min + F.softplus(rho)
    e_pos = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    e_neg = torch.tensor([[3.0, 0.5], [0.5, 3.0]])
    g = hinge_temp_logits(e_pos, e_neg, 0.5, torch.tensor(10.0), tau)
    F.cross_entropy(g, torch.tensor([0, 1])).backward()
    assert rho.grad is not None and rho.grad.abs().item() > 0
