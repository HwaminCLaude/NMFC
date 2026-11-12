import torch

from src.models import NMFCHead
from src.nmfc import local_energies, pairwise_sqdist, soft_neighbor_weights


def three_clusters(seed=0, n_per=24, m=4, gap=5.0):
    g = torch.Generator().manual_seed(seed)
    zs, ys = [], []
    centers = [torch.zeros(m), torch.full((m,), gap), torch.cat([torch.full((m // 2,), -gap),
                                                                 torch.zeros(m - m // 2)])]
    for c, center in enumerate(centers):
        zs.append(torch.randn(n_per, m, generator=g) * 0.5 + center)
        ys.append(torch.full((n_per,), c, dtype=torch.long))
    return torch.cat(zs), torch.cat(ys)


def test_bank_energies_match_direct_computation():
    """_bank_energies의 보수(complement) 항등식 E⁻가 정의대로 계산한 값과 일치해야 한다."""
    z, y = three_clusters()
    head = NMFCHead(n_ref=16)
    head.fit_bank(z, y)

    q = z[::5] + 0.3  # 뱅크에 없는 쿼리
    e_pos, e_neg = head._bank_energies(q)

    k, r, m = head.bank.shape
    flat = head.bank.reshape(k * r, m)
    y_flat = head.bank_classes.repeat_interleave(r)
    d2 = pairwise_sqdist(q, flat)
    pi_pos, pi_neg = soft_neighbor_weights(
        d2, y_flat, head.bank_classes, head.bank_sigma2, exclude_self=False
    )
    e_pos_ref, e_neg_ref = local_energies(d2, pi_pos, pi_neg)
    assert torch.allclose(e_pos, e_pos_ref, rtol=1e-4, atol=1e-5)
    assert torch.allclose(e_neg, e_neg_ref, rtol=1e-4, atol=1e-5)


def test_bank_path_consistent_with_batch_path():
    """뱅크 경로 예측이 배치 경로 로짓과 같은 답을 내야 한다 (잘 분리된 군집)."""
    z, y = three_clusters()
    head = NMFCHead(n_ref=16)

    g_batch, y_local = head.forward_batch(z, y)
    head.fit_bank(z, y)
    pred_bank = head.predict(z)

    assert (g_batch.argmax(1) == y_local).float().mean() == 1.0
    assert (pred_bank == y).float().mean() == 1.0


def test_ood_score_higher_for_far_points():
    z, y = three_clusters()
    head = NMFCHead(n_ref=16)
    head.fit_bank(z, y)
    near = z[:8] + 0.1
    far = torch.full((8, z.shape[1]), 50.0)
    assert head.ood_score(far).min() > head.ood_score(near).max()


def test_frozen_mu_sigma_do_not_depend_on_queries():
    """σ²·μ±는 뱅크에서 동결 — 어떤 쿼리를 넣어도 변하면 안 된다 (OOD 누수 방지)."""
    z, y = three_clusters()
    head = NMFCHead(n_ref=16)
    head.fit_bank(z, y)
    before = (head.bank_sigma2.clone(), head.bank_mu_pos.clone(), head.bank_mu_neg.clone())
    head.forward_bank(torch.randn(64, z.shape[1]) * 100)
    head.ood_score(torch.randn(64, z.shape[1]) * 100)
    assert torch.equal(before[0], head.bank_sigma2)
    assert torch.equal(before[1], head.bank_mu_pos)
    assert torch.equal(before[2], head.bank_mu_neg)
