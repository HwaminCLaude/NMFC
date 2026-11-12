"""비교 베이스라인 4종.

- LinearHead: 표준 Softmax CE (전역 선형 초평면)
- ArcFaceHead: 각도 마진 소프트맥스
- batch_hard_triplet_loss: Triplet (PK 배치 필요 — NMFC와 같은 샘플러 공유)
- ProtoHead: 학습형 프로토타입 — 클래스당 고정 중심점까지의 -거리² 로짓
- mfa_regularizer: Deep-MFA — MFA를 로짓이 아닌 보조 정규화항으로 쓰는 비교군
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.nmfc.affinity import pairwise_sqdist


class LinearHead(nn.Module):
    def __init__(self, embed_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, z):
        return self.fc(z)


class ArcFaceHead(nn.Module):
    """Additive angular margin: 학습 시 정답 클래스 각도에 margin을 더한다."""

    def __init__(self, embed_dim: int, n_classes: int, s: float = 30.0, margin: float = 0.2):
        super().__init__()
        self.s = s
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(n_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, z, y=None):
        cos = F.normalize(z) @ F.normalize(self.weight).T  # (N, K)
        if y is None:
            return self.s * cos  # 추론: 마진 없이
        theta = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
        cos_margin = torch.cos(theta + self.margin)
        onehot = F.one_hot(y, cos.shape[1]).bool()
        return self.s * torch.where(onehot, cos_margin, cos)


class ProtoHead(nn.Module):
    """logits = -||z - c_k||². 에피소드식 few-shot 프로토콜 대신 완전 지도에 공정한
    학습형 중심점 변형 — NMFC와 대비되는 '정적 중심점 거리 로짓'의 본질을 유지."""

    def __init__(self, embed_dim: int, n_classes: int):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(n_classes, embed_dim) * 0.05)

    def forward(self, z):
        return -pairwise_sqdist(z, self.prototypes)


def batch_hard_triplet_loss(z: torch.Tensor, y: torch.Tensor, margin: float = 0.3):
    """배치 내 가장 어려운 양성/음성으로 삼중항 손실. PK 배치(K>=2) 필요.

    FaceNet 원전대로 임베딩을 L2 정규화한다 — 정규화 없는 삼중항은 임베딩 전체를
    0으로 수축하는 자명해(loss=margin 고정)로 붕괴한다 (weight decay가 가속).
    """
    z = F.normalize(z)
    d = torch.cdist(z, z)
    n = len(z)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    same = (y[:, None] == y[None, :]) & ~eye
    diff = y[:, None] != y[None, :]
    if not same.any(dim=1).all():
        raise ValueError("모든 앵커에 양성 쌍이 필요합니다 (PK 샘플러 K>=2).")
    big = d.max().detach() + 1.0
    hardest_pos = d.masked_fill(~same, -1.0).max(dim=1).values
    hardest_neg = d.masked_fill(~diff, big).min(dim=1).values
    return F.relu(hardest_pos - hardest_neg + margin).mean()


def mfa_regularizer(
    z: torch.Tensor, y: torch.Tensor, k_intra: int = 5, k_inter: int = 5, eps: float = 1e-12
):
    """Deep-MFA 보조항: intrinsic 그래프(같은 클래스 kNN 거리합) /
    penalty 그래프(다른 클래스 최근접 마진 쌍 거리합). 작을수록 좋다."""
    d2 = pairwise_sqdist(z, z)
    n = len(z)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    same = (y[:, None] == y[None, :]) & ~eye
    diff = y[:, None] != y[None, :]
    big = torch.finfo(d2.dtype).max
    k_i = min(k_intra, n - 1)
    k_o = min(k_inter, n - 1)
    d_intra = d2.masked_fill(~same, big).topk(k_i, dim=1, largest=False).values
    d_inter = d2.masked_fill(~diff, big).topk(k_o, dim=1, largest=False).values
    intra = (d_intra * (d_intra < big / 2)).sum()
    inter = (d_inter * (d_inter < big / 2)).sum()
    return intra / inter.clamp_min(eps)


# ---- plan9: 강한 DML 베이스라인 (NormalizedSoftmax · Multi-Similarity · Proxy-Anchor) ----
# 전부 L2 정규화 임베딩(코사인) 위에서 동작 — 우리 shrb와 같은 규모(ResNet-50/CIFAR-100)의
# 검색 특화 학습법이라 "손실만 바꾼" 공정 비교 대상이다. 검색 평가는 헤드 제거 후 임베딩만 쓴다.


class NormSoftmaxHead(nn.Module):
    """정규화 소프트맥스 (Zhai & Wu, BMVC 2019). L2정규화 특징·가중치의 코사인 로짓 × 스케일."""

    def __init__(self, embed_dim: int, n_classes: int, scale: float = 20.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale

    def forward(self, z):
        return self.scale * (F.normalize(z) @ F.normalize(self.weight).T)


class ProxyAnchorHead(nn.Module):
    """Proxy-Anchor (Kim et al., CVPR 2020). 클래스별 학습 프록시. forward는 top1용 코사인 로짓."""

    def __init__(self, embed_dim: int, n_classes: int):
        super().__init__()
        self.proxies = nn.Parameter(torch.empty(n_classes, embed_dim))
        nn.init.kaiming_normal_(self.proxies, mode="fan_out")

    def forward(self, z):
        return F.normalize(z) @ F.normalize(self.proxies).T


def multi_similarity_loss(z, y, alpha=2.0, beta=50.0, base=0.5, eps=0.1):
    """Multi-Similarity loss (Wang et al., CVPR 2019). PK 배치(같은 클래스 K>=2) 필요.

    쌍 유사도 S=ẑ·ẑᵀ에서 하드 양/음성 소프트 마이닝 후 두 log-sum-exp 항을 합한다.
    """
    z = F.normalize(z)
    s = z @ z.T
    n = z.size(0)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    pos = (y[:, None] == y[None, :]) & ~eye
    neg = y[:, None] != y[None, :]
    # 마이닝 임계: 앵커별 가장 어려운 음성/양성 유사도
    neg_max = s.masked_fill(~neg, float("-inf")).max(1).values
    pos_min = s.masked_fill(~pos, float("inf")).min(1).values
    sel_pos = pos & (s < (neg_max[:, None] + eps))
    sel_neg = neg & (s > (pos_min[:, None] - eps))
    pos_term = torch.log1p((torch.exp(-alpha * (s - base)) * sel_pos).sum(1)) / alpha
    neg_term = torch.log1p((torch.exp(beta * (s - base)) * sel_neg).sum(1)) / beta
    loss = pos_term + neg_term
    valid = sel_pos.any(1) | sel_neg.any(1)
    return loss[valid].mean() if bool(valid.any()) else loss.mean()


def proxy_anchor_loss(z, y, proxies, margin=0.1, alpha=32.0):
    """Proxy-Anchor loss (Kim et al., CVPR 2020). proxies: (C, d) 학습 파라미터."""
    z = F.normalize(z)
    p = F.normalize(proxies)
    c = p.size(0)
    s = z @ p.T  # (N, C) 코사인
    pos = F.one_hot(y, c).bool()
    pos_exp = (torch.exp(-alpha * (s - margin)) * pos).sum(0)  # (C,)
    neg_exp = (torch.exp(alpha * (s + margin)) * ~pos).sum(0)
    with_pos = pos.any(0)
    n_with_pos = with_pos.sum().clamp(min=1)
    pos_term = (torch.log1p(pos_exp) * with_pos).sum() / n_with_pos
    neg_term = torch.log1p(neg_exp).sum() / c
    return pos_term + neg_term
