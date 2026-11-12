"""국소 기하 에너지 E±, 경험적 평균 스케일링 μ±, 국소 기하 로짓 g."""

import torch


def local_energies(
    d2: torch.Tensor, pi_pos: torch.Tensor, pi_neg: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """E±_k(z_i) = Σ_j π±_ij(k) · ‖z_i - z_j‖²  →  (N, P)."""
    e_pos = torch.einsum("pnm,nm->np", pi_pos, d2)
    e_neg = torch.einsum("pnm,nm->np", pi_neg, d2)
    return e_pos, e_neg


def mean_scale(
    e_pos: torch.Tensor, e_neg: torch.Tensor, y_local: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """경험적 평균 스케일링 μ± — 정답 클래스 에너지의 배치 평균.

    detach 필수: μ를 통해 역전파하면 임베딩을 전체 스케일만 줄이는 편법으로
    손실이 내려가 기하 구조 학습이 멈춘다 (준정적 유도와도 일치).
    """
    idx = y_local.view(-1, 1)
    mu_pos = e_pos.gather(1, idx).mean().detach().clamp_min(1e-12)
    mu_neg = e_neg.gather(1, idx).mean().detach().clamp_min(1e-12)
    return mu_pos, mu_neg


def geometric_logits(
    e_pos: torch.Tensor,
    e_neg: torch.Tensor,
    mu_pos: torch.Tensor,
    mu_neg: torch.Tensor,
    lam: float = 0.5,
) -> torch.Tensor:
    """국소 기하 로짓 g_k = -E⁺_k/μ⁺ + λ·E⁻_k/μ⁻  →  (N, P)."""
    return -e_pos / mu_pos + lam * (e_neg / mu_neg)
