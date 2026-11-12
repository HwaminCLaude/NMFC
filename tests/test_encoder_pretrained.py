"""from-scratch 백본 테스트 — pretrained=False 랜덤 초기화 (plan5.md).

다운로드가 필요한 pretrained=True는 테스트하지 않는다(CI 속도·오프라인). weights=None 경로만 검증.
"""

import torch

from src.models.encoders import ResNet50Encoder


def test_pretrained_false_builds_and_forwards():
    """pretrained=False는 다운로드 없이 빌드되고 (N, embed_dim)을 낸다."""
    enc = ResNet50Encoder(embed_dim=128, pretrained=False).eval()
    with torch.no_grad():
        out = enc(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 128)


def test_pretrained_false_is_random_init():
    """랜덤 초기화 — conv1 가중치가 상수가 아니고, 시드 다르면 값도 다르다."""
    torch.manual_seed(0)
    a = ResNet50Encoder(embed_dim=128, pretrained=False)
    w = a.net.conv1.weight
    assert w.std().item() > 0.0            # 상수 아님(실제 초기화됨)
    torch.manual_seed(1)
    b = ResNet50Encoder(embed_dim=128, pretrained=False)
    assert not torch.allclose(w, b.net.conv1.weight)  # 결정적 상수 아님
