"""임베딩 인코더 f_theta — 2D 합성용 MLP와 이미지용 ResNet-50.

주의: 임베딩 z를 L2 정규화하지 않는다. NMFC 에너지는 원시 유클리드 거리를
사용하며, 정규화하면 거리 스케일 정보가 사라진다.
"""

import torch.nn as nn


class EncoderMLP(nn.Module):
    """합성 2D 데이터용. embed_dim=2로 두면 임베딩 공간을 그대로 그릴 수 있다."""

    def __init__(self, in_dim: int = 2, hidden: tuple[int, ...] = (64, 64), embed_dim: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResNet50Encoder(nn.Module):
    """ImageNet 사전학습 ResNet-50, fc를 m차원 임베딩으로 교체 (기본 m=128)."""

    def __init__(self, embed_dim: int = 128, pretrained: bool = True):
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = resnet50(weights=weights)
        net.fc = nn.Linear(net.fc.in_features, embed_dim)
        self.net = net

    def forward(self, x):
        return self.net(x)
