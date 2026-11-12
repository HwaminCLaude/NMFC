"""자동 위상 전이(APT) 컨트롤러. torch 의존 없는 순수 파이썬 상태기계.

매 에폭 말에 Fisher ratio R_t를 update()로 넣으면 현재 phase를 돌려준다.
Φ_t = 최근 window개 R의 이동평균. |ΔΦ|/Φ < delta가 patience 에폭 연속이면
LINEAR → NONLINEAR로 단방향 전환한다 (이중 단계 프로토콜).
가드: min_warmup(사전학습 백본의 조기 안정 대응), max_warmup(δ 미발동 폴백).
"""

LINEAR = "linear"
NONLINEAR = "nonlinear"


class APTController:
    def __init__(
        self,
        window: int = 5,
        delta: float = 1e-4,
        patience: int = 5,
        min_warmup: int = 5,
        max_warmup: int = 30,
    ):
        self.window = window
        self.delta = delta
        self.patience = patience
        self.min_warmup = min_warmup
        self.max_warmup = max_warmup
        self.ratios: list[float] = []
        self.phis: list[float] = []
        self.phase = LINEAR
        self.transition_epoch: int | None = None
        self.transition_reason: str | None = None
        self._streak = 0

    def update(self, ratio: float) -> str:
        self.ratios.append(float(ratio))
        phi = sum(self.ratios[-self.window :]) / len(self.ratios[-self.window :])
        self.phis.append(phi)
        if self.phase == NONLINEAR:
            return self.phase
        epoch = len(self.ratios)
        if epoch >= 2:
            rel = abs(phi - self.phis[-2]) / max(abs(phi), 1e-12)
            self._streak = self._streak + 1 if rel < self.delta else 0
        if epoch >= self.min_warmup and self._streak >= self.patience:
            self._fire(epoch, "converged")
        elif epoch >= self.max_warmup:
            self._fire(epoch, "max_warmup")
        return self.phase

    def _fire(self, epoch: int, reason: str) -> None:
        self.phase = NONLINEAR
        self.transition_epoch = epoch
        self.transition_reason = reason

    @property
    def history(self) -> dict:
        return {
            "fisher_ratio": self.ratios,
            "phi": self.phis,
            "transition_epoch": self.transition_epoch,
            "transition_reason": self.transition_reason,
        }


class FixedTransition(APTController):
    """절제 실험용: Φ를 기록하되 지정한 에폭에서 무조건 전환."""

    def __init__(self, fixed_epoch: int, window: int = 5):
        super().__init__(window=window)
        self.fixed_epoch = fixed_epoch

    def update(self, ratio: float) -> str:
        self.ratios.append(float(ratio))
        phi = sum(self.ratios[-self.window :]) / len(self.ratios[-self.window :])
        self.phis.append(phi)
        if self.phase == LINEAR and len(self.ratios) >= self.fixed_epoch:
            self._fire(len(self.ratios), "fixed")
        return self.phase


class NoWarmup(APTController):
    """절제 실험용: 워밍업 없이 처음부터 비선형 단계."""

    def __init__(self, window: int = 5):
        super().__init__(window=window)
        self.phase = NONLINEAR
        self.transition_epoch = 0
        self.transition_reason = "none"

    def update(self, ratio: float) -> str:
        self.ratios.append(float(ratio))
        phi = sum(self.ratios[-self.window :]) / len(self.ratios[-self.window :])
        self.phis.append(phi)
        return self.phase
