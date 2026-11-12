"""탐색용 국면 I 손실 -J와 ACS 컨트롤러 — **채택하지 않은 변형이다.**

  ⚠ 채택형이 아니다. 채택한 국면 I 손실은 역수형
    L_I = 1/(J+ε) 이고, 그 구현은 이 파일이 아니라 `src/nmfc/losses.py:linear_mfa_loss` 다.
    이 파일의 -J 계열은 실측으로 **기각된** 변형이다(-J는 하유계가 없고,
    임베딩 노름 팽창이 관측됐다). 실측 대조는 다음과 같다
    (c100v_shjb: embed_norm 16.5 → 395 → 3.9, top-1 0.845 → 0.787).

아래 서술은 이 변형을 도입할 당시의 설계 의도이며, 실측이 그 의도를 기각했다는 점을 유의하라.

v1은 워밍업 손실을 L_LMF = 1/J(역수)로 두어 J가 커질수록 경사가 J⁻²로 소멸했다.
이 변형은 부호 반전 -J를 직접 최소화해 경사 소멸을 제거하려 했다(하유계는 잃지만, 국면 II의
힌지 마진·가중치 감쇠·T_max 하드캡의 3중 방어선으로 팽창을 다스린다는 v3 설계 가정).
→ 결과: 팽창을 다스리지 못했고, 역수형을 유지했다.

컨트롤러는 apt.py:APTController와 동일 덕타입이되, (i) 비율이 아니라 J를 넣고,
(ii) 상대변화 분모를 max(Φ_{u-1}, ε_Φ)로 가드하며(J≥0 → Φ≥0), (iii) 램프 계수 α_t를 제공한다.
LINEAR/NONLINEAR 상수는 apt.py에서 import(중복 정의 금지).
"""

import torch

from src.nmfc.apt import LINEAR, NONLINEAR
from src.nmfc.losses import scatter_matrices


def fisher_j(
    z: torch.Tensor, y: torch.Tensor, eta_rel: float = 1e-3, sphere: bool = False
) -> torch.Tensor:
    """피셔 판별 기준 J = Tr((S_W + ηI)⁻¹ S_B). grad 보유 스칼라 텐서.

    η = eta_rel·Tr(S_W)/d (상대 릿지, detach). linalg.solve로 명시적 역행렬 회피.
    losses.py:linear_mfa_loss가 1/(J+ε)이므로 fisher_j ≈ 1/linear_mfa_loss(동일 J).

    sphere=True (v4): 산포를 원 임베딩이 아니라 단위 구면에 정규화한
    ẑ_i = z_i/‖z_i‖ 위에서 계산한다. J가 임베딩 스케일에 불변(z→az가 ẑ 불변)이 되어,
    무제약 -J의 임베딩 팽창 자유도가 척도 불변성으로 구조적으로 닫힌다. 기본 False → v3와 동일.
    """
    if sphere:
        z = z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)
    s_w, s_b = scatter_matrices(z, y)
    d = z.shape[1]
    s_w = 0.5 * (s_w + s_w.T)  # 수치 대칭화
    eta = (eta_rel * s_w.diagonal().sum() / d).detach().clamp_min(1e-8)
    eye = torch.eye(d, device=z.device, dtype=z.dtype)
    return torch.linalg.solve(s_w + eta * eye, s_b).diagonal().sum()


def neg_j_loss(
    z: torch.Tensor,
    y: torch.Tensor,
    eta_rel: float = 1e-3,
    sphere: bool = False,
    kappa: float | None = None,
) -> torch.Tensor:
    """국면 I 손실 L_lin = -J. v1 역수형의 J⁻² 경사 소멸을 제거한다.

    - 기본: -J (원 공간, v3).
    - sphere=True: -J(단위 구면, v4 기본) — fisher_j(sphere=True).
    - kappa 지정: 원 공간 포화형 trace-ratio -(J/(J+κ)) (v4 절제, J→∞에서 -1로 포화).
      κ는 전환 이전 J 중앙값에 앵커(호출부가 결정). sphere와 동시 지정 금지.
    """
    assert not (sphere and kappa is not None), "sphere와 trace-ratio(kappa)는 동시 지정 불가"
    j = fisher_j(z, y, eta_rel, sphere=sphere)
    if kappa is not None:
        return -(j / (j + kappa))
    return -j


class ACSController:
    """J의 이동평균 Φ 정체를 감지해 국면 I→II로 단방향 전환(래치)하는 상태기계.

    __init__ assert: window + patience <= t_min (전환 판정에 필요한 가장 이른 항 Φ_{t-P}가
    정의되려면 t ≥ W+P여야 하므로, 최소 학습 에폭 T_min이 이를 포함해야 한다).
    """

    def __init__(
        self,
        window: int = 5,
        patience: int = 5,
        delta: float = 1e-4,
        t_min: int = 10,
        t_max: int = 30,
        ramp: int = 5,
        eps_phi: float = 1e-12,
    ):
        assert window + patience <= t_min, (
            f"ACS 일관성 위반: window+patience={window + patience} > t_min={t_min} "
            f"(W+P ≤ T_min 이어야 함)"
        )
        self.window = window
        self.patience = patience
        self.delta = delta
        self.t_min = t_min
        self.t_max = t_max
        self.ramp = ramp
        self.eps_phi = eps_phi
        self.js: list[float] = []
        self.phis: list[float] = []
        self.phase = LINEAR
        self.transition_epoch: int | None = None
        self.transition_reason: str | None = None
        self._streak = 0

    def update(self, j_value: float) -> str:
        self.js.append(float(j_value))
        phi = sum(self.js[-self.window :]) / len(self.js[-self.window :])
        self.phis.append(phi)
        if self.phase == NONLINEAR:
            return self.phase
        epoch = len(self.js)
        if epoch >= 2:
            rel = abs(phi - self.phis[-2]) / max(self.phis[-2], self.eps_phi)
            self._streak = self._streak + 1 if rel < self.delta else 0
        if epoch >= self.t_min and self._streak >= self.patience:
            self._fire(epoch, "converged")
        elif epoch >= self.t_max:
            self._fire(epoch, "max_warmup")
        return self.phase

    def _fire(self, epoch: int, reason: str) -> None:
        self.phase = NONLINEAR
        self.transition_epoch = epoch
        self.transition_reason = reason

    def alpha(self, epoch: int) -> float:
        """혼합 계수 α_t = min(1, (t - t_c)/R). 전환 전이면 0.0, ramp=0이면 전환 후 1.0."""
        if self.transition_epoch is None or epoch <= self.transition_epoch:
            return 0.0
        if self.ramp == 0:
            return 1.0
        return min(1.0, (epoch - self.transition_epoch) / self.ramp)

    @property
    def history(self) -> dict:
        return {
            "j": self.js,
            "phi": self.phis,
            "transition_epoch": self.transition_epoch,
            "transition_reason": self.transition_reason,
        }
