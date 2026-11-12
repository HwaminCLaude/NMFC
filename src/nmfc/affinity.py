"""쌍별 거리, 커널 대역폭 자동 튜닝, 소프트 이웃 가중치 π.

π는 수식의 a/(Σa+ε) 대신 마스크드 softmax로 계산한다 (ε→0에서 수학적 동치).
문자 그대로 구현하면 어떤 클래스의 이웃이 전부 멀 때 exp(-d²/σ²)가 언더플로해
π→0, E⁺→0이 되어 '가장 먼 클래스가 최고 로짓'이 되는 침묵 버그가 생긴다.
마스크드 softmax는 E⁺를 클래스까지의 soft-min 거리로 유지해 무조건 안정적이다.
"""

import torch


def pairwise_sqdist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """제곱 유클리드 거리 행렬 (N, M)."""
    return torch.cdist(a, b).pow(2)


def median_sigma2(d2: torch.Tensor, exclude_diag: bool = True) -> torch.Tensor:
    """σ² 자동 튜닝: 배치 쌍별 제곱 거리의 중앙값.

    제곱은 단조이므로 median(d)² = median(d²) — d²에서 바로 중앙값을 취한다.
    대역폭은 스케일 휴리스틱이라 그래디언트를 흘리지 않는다 (detach).
    """
    if exclude_diag and d2.dim() == 2 and d2.shape[0] == d2.shape[1]:
        n = d2.shape[0]
        mask = ~torch.eye(n, dtype=torch.bool, device=d2.device)
        vals = d2[mask]
    else:
        vals = d2.reshape(-1)
    return vals.detach().median().clamp_min(1e-12)


def soft_neighbor_weights(
    d2: torch.Tensor,
    y_col: torch.Tensor,
    classes: torch.Tensor,
    sigma2: torch.Tensor,
    exclude_self: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """소프트 이웃 가중치 π± (P, N, M).

    Args:
        d2: (N, M) 제곱 거리 — 행은 앵커, 열은 이웃 후보.
        y_col: (M,) 열(이웃) 레이블.
        classes: (P,) 에너지를 계산할 클래스 목록.
        sigma2: 커널 대역폭 σ² (스칼라 텐서).
        exclude_self: 정방 d2일 때 대각(자기쌍)을 이웃에서 제외. 배치 내 학습에서는
            필수 — 빠뜨리면 자기 자신과의 거리 0이 지배해 E⁺≈0으로 붕괴한다.

    Returns:
        (pi_pos, pi_neg): 각 (P, N, M), 마지막 축 합=1.
        해당 클래스 이웃이 하나도 없는 행은 0으로 채워진다 (호출부에서 방지할 것 —
        PK 샘플러가 P≥2, K≥2를 보장한다).
    """
    logk = -d2 / sigma2
    if exclude_self and d2.shape[0] == d2.shape[1]:
        eye = torch.eye(d2.shape[0], dtype=torch.bool, device=d2.device)
        logk = logk.masked_fill(eye, float("-inf"))
    member = y_col[None, :] == classes[:, None]  # (P, M)
    logits = logk.unsqueeze(0).expand(len(classes), -1, -1)  # (P, N, M) — view, 복사 없음
    pi_pos = torch.softmax(logits.masked_fill(~member[:, None, :], float("-inf")), dim=-1)
    pi_neg = torch.softmax(logits.masked_fill(member[:, None, :], float("-inf")), dim=-1)
    # 이웃이 전무한 행(전부 -inf)은 softmax가 NaN을 내므로 0으로 정리
    pi_pos = torch.nan_to_num(pi_pos, nan=0.0)
    pi_neg = torch.nan_to_num(pi_neg, nan=0.0)
    return pi_pos, pi_neg
