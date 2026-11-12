"""NMFC NLL 손실, 선형 MFA 워밍업 손실 L_LMF, 산포 행렬과 Fisher ratio."""

import torch
import torch.nn.functional as F


def nmfc_loss(g: torch.Tensor, y_local: torch.Tensor) -> torch.Tensor:
    """음의 로그가능도. softmax는 cross_entropy 내부의 LSE로 안정 처리."""
    return F.cross_entropy(g, y_local)


def scatter_matrices(z: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """클래스 내(S_W)·클래스 간(S_B) 산포 행렬, 각 (m, m). 1/n 정규화."""
    classes, y_local = torch.unique(y, return_inverse=True)
    n, m = z.shape
    p = len(classes)
    counts = torch.bincount(y_local, minlength=p).to(z.dtype)
    sums = z.new_zeros(p, m).index_add_(0, y_local, z)
    mu_c = sums / counts[:, None]
    zc = z - mu_c[y_local]
    s_w = zc.T @ zc / n
    d = mu_c - z.mean(0, keepdim=True)
    s_b = (d * counts[:, None]).T @ d / n
    return s_w, s_b


def fisher_ratio(z: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Φ의 재료가 되는 단순 trace 비율 Tr(S_B)/Tr(S_W)."""
    s_w, s_b = scatter_matrices(z, y)
    return s_b.diagonal().sum() / (s_w.diagonal().sum() + eps)


def linear_mfa_loss(
    z: torch.Tensor, y: torch.Tensor, eta_rel: float = 1e-3, eps: float = 1e-6
) -> torch.Tensor:
    """L_LMF = Tr((S_W + ηI)^{-1} S_B)^{-1} — Phase I 워밍업 손실.

    η는 상대 릿지 η = eta_rel·Tr(S_W)/m (임베딩 스케일에 불변, detach).
    명시적 역행렬 대신 linalg.solve. 초기에 F≈0이라 1/F가 폭발할 수 있으므로
    분모에 ε — 호출부는 grad clip과 함께 쓸 것.
    """
    s_w, s_b = scatter_matrices(z, y)
    m = z.shape[1]
    s_w = 0.5 * (s_w + s_w.T)  # 수치 대칭화
    eta = (eta_rel * s_w.diagonal().sum() / m).detach().clamp_min(1e-8)
    eye = torch.eye(m, device=z.device, dtype=z.dtype)
    f_val = torch.linalg.solve(s_w + eta * eye, s_b).diagonal().sum()
    return 1.0 / (f_val + eps)
