"""plan9 강한 DML 손실 단위 테스트 — 완전분리 임베딩에서 손실 최소·grad 흐름 (CPU, 수 초)."""

import torch

from src.models.baselines import (
    NormSoftmaxHead,
    ProxyAnchorHead,
    multi_similarity_loss,
    proxy_anchor_loss,
)


def _separated(n_class=4, per=8, dim=16, seed=0):
    """클래스별 직교 방향 + 소량 노이즈 = 완전분리 임베딩."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.eye(n_class, dim)
    y = torch.arange(n_class).repeat_interleave(per)
    z = centers[y] * 5.0 + 0.01 * torch.randn(n_class * per, dim, generator=g)
    return z, y


def test_ms_loss_low_on_separated_high_on_mixed():
    z, y = _separated()
    sep = multi_similarity_loss(z, y).item()
    y_shuf = y[torch.randperm(len(y), generator=torch.Generator().manual_seed(1))]
    mixed = multi_similarity_loss(z, y_shuf).item()
    assert sep < mixed, (sep, mixed)
    assert sep >= 0.0


def test_ms_loss_grad_flows():
    z, y = _separated()
    z = z.requires_grad_(True)
    multi_similarity_loss(z, y).backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_proxy_anchor_loss_grad_and_finite():
    z, y = _separated()
    z = z.requires_grad_(True)
    proxies = torch.randn(4, 16, requires_grad=True)
    loss = proxy_anchor_loss(z, y, proxies)
    loss.backward()
    assert torch.isfinite(loss)
    assert z.grad is not None and proxies.grad is not None


def test_normsoftmax_head_shape_and_ce():
    z, y = _separated()
    head = NormSoftmaxHead(16, 4, scale=20.0)
    logits = head(z)
    assert logits.shape == (len(y), 4)
    assert torch.isfinite(torch.nn.functional.cross_entropy(logits, y))
    # 코사인 × scale이므로 |logit| <= scale (헤드는 미학습이라 정답 분류는 학습 후에만)
    assert logits.abs().max() <= 20.0 + 1e-4


def test_normsoftmax_head_learns_on_separated():
    """짧은 최적화로 정답 방향 정렬 — 손실이 실제로 감소하는지."""
    z, y = _separated()
    head = NormSoftmaxHead(16, 4, scale=20.0)
    opt = torch.optim.SGD(head.parameters(), lr=1.0)
    ce = torch.nn.functional.cross_entropy
    first = ce(head(z), y).item()
    for _ in range(50):
        opt.zero_grad()
        loss = ce(head(z), y)
        loss.backward()
        opt.step()
    assert loss.item() < first
    assert (head(z).argmax(1) == y).float().mean() > 0.9  # 학습 후엔 분류 가능


def test_proxyanchor_head_cosine_range():
    z, _ = _separated()
    head = ProxyAnchorHead(16, 4)
    out = head(z)
    assert out.shape == (len(z), 4)
    assert out.abs().max() <= 1.0 + 1e-5  # 코사인
