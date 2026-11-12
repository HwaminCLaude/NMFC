import pytest
import torch

from src.models.banks import OnlineClassBank
from src.models.heads_v3 import VariantNMFCHead


def toy_pk(p=4, k=4, dim=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    zs, ys = [], []
    for c in range(p):
        center = 5.0 * torch.randn(dim, generator=g)
        zs.append(center + 0.3 * torch.randn(k, dim, generator=g))
        ys.append(torch.full((k,), c, dtype=torch.long))
    return torch.cat(zs), torch.cat(ys)


def make_head(energy="softmin", k1=3, k2=8, n_coreset=8):
    return VariantNMFCHead(
        energy=energy, logit="hinge_temp", lam=0.5, k1=k1, k2=k2, tau_min=0.01,
        m_quantile=0.75, n_coreset=n_coreset, n_ref=8, detach_pi=False,
    )


def test_forward_batch_finite_softmin():
    z, y = toy_pk()
    head = make_head("softmin")
    head.set_margin(torch.rand(16) + 1.0)
    g, y_local = head.forward_batch(z, y)
    assert g.shape == (16, 4) and torch.isfinite(g).all()
    assert int(y_local.max()) < 4


def test_forward_batch_finite_mean():
    z, y = toy_pk()
    head = make_head("mean")
    head.set_margin(torch.rand(16) + 1.0)
    g, _ = head.forward_batch(z, y)
    assert g.shape == (16, 4) and torch.isfinite(g).all()


def test_tau_init_and_lower_bound():
    head = make_head()
    assert abs(head.tau.item() - 1.0) < 1e-4  # τ₀ = 1.0
    with torch.no_grad():
        head.rho.fill_(-100.0)
    # ρ→−∞ 극한에서 τ→τ_min (float32 표현 오차 허용); 하한 아래로는 못 감
    assert head.tau.item() >= head.tau_min - 1e-6


def test_coreset_bank_infer_and_freeze():
    z, y = toy_pk(p=4, k=20, dim=8)
    head = make_head("softmin")
    head.set_margin(torch.rand(80) + 1.0)
    head.fit_coreset_bank(z, y)
    assert torch.isfinite(head.bank_sigma2) and torch.isfinite(head.bank_tau)
    assert torch.isfinite(head.bank_m)
    zt = z[:10]
    assert head.forward_bank(zt).shape == (10, 4)
    assert (head.predict(zt) == y[:10]).float().mean() > 0.5  # 학습셋 근처 자기 클래스
    ood = head.ood_score(zt)
    assert ood.shape == (10,) and torch.isfinite(ood).all()


def test_ood_score_grows_far():
    """명제 1: far-OOD에서 s(x) = min_k E⁺_k 증가."""
    z, y = toy_pk(p=4, k=20, dim=8)
    head = make_head("softmin")
    head.set_margin(torch.rand(80) + 1.0)
    head.fit_coreset_bank(z, y)
    near = head.ood_score(z[:20]).mean()
    far = head.ood_score(z[:20] + 100.0).mean()
    assert far > near


def test_bank_state_roundtrip():
    z, y = toy_pk(p=4, k=20, dim=8)
    head = make_head("softmin")
    head.set_margin(torch.rand(80) + 1.0)
    head.fit_coreset_bank(z, y)
    zt = z[:10]
    logits0 = head.forward_bank(zt)
    head2 = make_head("softmin")
    head2.load_bank_state(head.bank_state(), device="cpu")
    assert torch.allclose(logits0, head2.forward_bank(zt), atol=1e-5)
    assert head2.energy == "softmin"


def test_fit_coreset_asserts_ncoreset_ge_k1():
    z, y = toy_pk(p=4, k=20, dim=8)
    head = make_head("softmin", k1=10, n_coreset=8)  # n_coreset < k1
    head.set_margin(torch.rand(80) + 1.0)
    with pytest.raises(AssertionError):
        head.fit_coreset_bank(z, y)


def test_forward_online_and_loo():
    z, y = toy_pk(p=4, k=8, dim=8)  # 32 samples
    idx = torch.arange(32)
    bank = OnlineClassBank(n_classes=4, capacity=8, dim=8, device=torch.device("cpu"))
    bank.charge(z, y, idx)
    head = make_head("softmin", k1=3, k2=8)
    head.set_margin(torch.rand(32) + 1.0)
    g, y_local = head.forward_online(z, y, bank, idx)
    assert g.shape == (32, 4) and torch.isfinite(g).all()
    assert torch.equal(y_local, y)
    _, _, idx_ref = bank.reference()
    excl = head._loo_cols(idx, idx_ref)
    assert (excl >= 0).all()  # 모든 쿼리 태그가 뱅크에 존재
    assert idx_ref[excl[0]].item() == idx[0].item()


def test_forward_online_grad_flows_to_query():
    z, y = toy_pk(p=4, k=8, dim=8)
    idx = torch.arange(32)
    bank = OnlineClassBank(4, 8, 8, torch.device("cpu"))
    bank.charge(z, y, idx)
    head = make_head("softmin", k1=3, k2=8)
    head.set_margin(torch.rand(32) + 1.0)
    zq = z.clone().requires_grad_(True)
    g, y_local = head.forward_online(zq, y, bank, idx)
    torch.nn.functional.cross_entropy(g, y_local).backward()
    assert zq.grad is not None and torch.isfinite(zq.grad).all()
    # 뱅크 참조는 detach → grad 없음
    assert not bank.reference()[0].requires_grad
