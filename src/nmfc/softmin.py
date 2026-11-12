"""v3 수식 — soft-min(log-sum-exp) 국소 에너지, kNN 이웃 집합, 힌지·온도 로짓.

v1의 정규화 π 가중평균(affinity.py/energy.py)과 병존하는 **새 수식 모듈**이다(설계 규칙:
v1 수학 모듈 불변, v3는 새 파일로 분리). soft-min 에너지는 비정규화 log-sum-exp이라
안전장치 ε 없이도 수치적으로 안정하며(max-shift 내장), 이웃 집합 선택은 stop-gradient로 처리해
경사 분해가 구성상 정확한 식이 되게 한다.

침묵 실패 방지:
- topk(이웃 선택)는 d2.detach()에서 — 그래디언트는 gather된 거리로만 흐른다(정지 경사).
- leave-one-out(자기/뱅크 태그 열 제외)은 masked_fill(big)로 후보에서 배제.
- 후보 수 < k이면 topk가 big 채운 열을 골라 오염되므로, 호출부가 k ≤ 후보수를 assert로 보장한다.
"""

import torch


def knn_neighbor_sets(
    d2: torch.Tensor,
    y_col: torch.Tensor,
    classes: torch.Tensor,
    k1: int,
    k2: int,
    exclude_cols: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """클래스별 k₁-NN(동일 클래스 N_k)·k₂-NN(타 클래스 M_k) 이웃 열 인덱스.

    Args:
        d2: (N, M) 제곱 거리 — 행=쿼리, 열=참조 후보.
        y_col: (M,) 참조 레이블.
        classes: (P,) 에너지를 계산할 대상 클래스.
        k1, k2: 동일/타 클래스 최근접 이웃 수.
        exclude_cols: (N,) 각 쿼리가 제외할 열 인덱스(leave-one-out). 원소 -1이면 제외 없음.
            배치 내 참조에서는 자기 행 인덱스, 온라인 뱅크에서는 같은 sample-index 슬롯.

    Returns:
        (idx_pos (P,N,k1), idx_neg (P,N,k2)) long — d2 열 방향 인덱스.
        선택(topk)은 d2.detach()에서 수행(stop-gradient).
    """
    n, m = d2.shape
    p = len(classes)
    dsel = d2.detach()
    big = torch.finfo(dsel.dtype).max
    idx_pos = d2.new_empty(p, n, k1, dtype=torch.long)
    idx_neg = d2.new_empty(p, n, k2, dtype=torch.long)

    excl_rows = excl_cols = None
    if exclude_cols is not None:
        valid = exclude_cols >= 0
        excl_rows = torch.arange(n, device=d2.device)[valid]
        excl_cols = exclude_cols[valid]

    for j, c in enumerate(classes):
        same = y_col == c  # (M,)
        d_pos = dsel.masked_fill(~same[None, :], big)  # 타 클래스 열 제외
        d_neg = dsel.masked_fill(same[None, :], big)   # 동일 클래스 열 제외
        if excl_rows is not None:
            d_pos[excl_rows, excl_cols] = big
            d_neg[excl_rows, excl_cols] = big
        idx_pos[j] = d_pos.topk(k1, dim=-1, largest=False).indices
        idx_neg[j] = d_neg.topk(k2, dim=-1, largest=False).indices
    return idx_pos, idx_neg


def softmin_energies(d2: torch.Tensor, idx: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
    """비정규화 soft-min 에너지 E = -σ²·logsumexp(-gather(d2, idx)/σ², dim=-1)  →  (N, P).

    Args:
        d2: (N, M) 제곱 거리 (grad 보유).
        idx: (P, N, k) 이웃 열 인덱스 (knn_neighbor_sets 산출, detach된 인덱스).
        sigma2: 커널 대역폭 σ² (스칼라 텐서, detach).

    logsumexp가 max-shift를 내부 처리하므로 원거리 언더플로가 없다(명제 1의 하한과 정합).
    """
    p, n, _ = idx.shape
    d2_exp = d2.unsqueeze(0).expand(p, n, -1)  # (P, N, M) — 브로드캐스트 뷰(복사 없음)
    gathered = torch.gather(d2_exp, 2, idx)     # (P, N, k)
    lse = torch.logsumexp(-gathered / sigma2, dim=-1)  # (P, N)
    e = -sigma2 * lse
    return e.transpose(0, 1).contiguous()  # (N, P)


def hinge_temp_logits(
    e_pos: torch.Tensor,
    e_neg: torch.Tensor,
    lam: float,
    m: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """힌지·온도 로짓 g_k = (-E⁺_k + λ·min(E⁻_k, m)) / τ  →  (N, P).

    torch.minimum(E⁻, m)은 E⁻ > m인 표본에서 척력 그래디언트를 정확히 0으로 끊는다
    (지시함수 1[E⁻<m]). m·τ는 스칼라 텐서.
    """
    return (-e_pos + lam * torch.minimum(e_neg, m)) / tau
