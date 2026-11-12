"""NMFC 헤드 — 학습(배치 내)과 추론(ReferenceBank)의 두 경로.

수학은 src/nmfc를 호출하고, 이 클래스는 상태(λ, 뱅크, 동결된 σ²·μ±)만 관리한다.

추론 원칙 (침묵 실패 방지):
- σ²·μ±는 뱅크에서 한 번 동결한다. 테스트 배치에서 재계산하면 로짓이 배치
  크기에 의존하고 OOD 통계가 테스트 분포로 새어 들어간다.
- 뱅크 자신을 쿼리로 μ±를 잴 때는 자기 슬롯을 제외한다 (d²=0이 softmax를
  지배해 μ⁺가 0 근처로 왜곡되는 것을 방지).
- E⁻는 (C,K,K·R) 텐서를 만들지 않고 전체 합 - 자기 클래스 합의 보수 항등식으로
  구한다. 한 클래스가 가중치를 독점할 때의 소거 오차를 피하려고 float64 부분합.
"""

import torch
import torch.nn as nn

from src.nmfc.affinity import median_sigma2, pairwise_sqdist, soft_neighbor_weights
from src.nmfc.energy import geometric_logits, local_energies, mean_scale


class NMFCHead(nn.Module):
    def __init__(self, lam: float = 0.5, n_ref: int = 64, detach_pi: bool = False):
        super().__init__()
        self.lam = lam
        self.n_ref = n_ref
        self.detach_pi = detach_pi
        # ReferenceBank 상태 (fit_bank에서 채움)
        self.bank: torch.Tensor | None = None  # (K, R, m)
        self.bank_classes: torch.Tensor | None = None  # (K,)
        self.bank_sigma2: torch.Tensor | None = None
        self.bank_mu_pos: torch.Tensor | None = None
        self.bank_mu_neg: torch.Tensor | None = None

    # ---------- 학습 경로: 배치 내 에너지 ----------

    def forward_batch(self, z: torch.Tensor, y: torch.Tensor):
        """배치 임베딩 (B,m) → (로짓 (B,P), 배치 내 로컬 레이블 (B,)).

        로짓은 배치에 등장한 P개 클래스에 대해서만 계산한다 (sampled softmax).
        PK 샘플러가 P>=2, K>=2를 보장해야 한다.
        """
        classes, y_local = torch.unique(y, return_inverse=True)
        if len(classes) < 2:
            raise ValueError("NMFC 배치에는 클래스가 2개 이상 필요합니다 (PK 샘플러 사용).")
        d2 = pairwise_sqdist(z, z)
        sigma2 = median_sigma2(d2)
        pi_pos, pi_neg = soft_neighbor_weights(d2, y, classes, sigma2, exclude_self=True)
        if self.detach_pi:
            pi_pos, pi_neg = pi_pos.detach(), pi_neg.detach()
        e_pos, e_neg = local_energies(d2, pi_pos, pi_neg)
        mu_pos, mu_neg = mean_scale(e_pos, e_neg, y_local)
        g = geometric_logits(e_pos, e_neg, mu_pos, mu_neg, self.lam)
        return g, y_local

    # ---------- 추론 경로: ReferenceBank ----------

    @torch.no_grad()
    def fit_bank(self, z_train: torch.Tensor, y_train: torch.Tensor, seed: int = 42) -> None:
        """train 임베딩에서 클래스당 최대 n_ref개를 샘플해 뱅크를 만들고 σ²·μ±를 동결."""
        classes = torch.unique(y_train)  # 정렬됨
        if len(classes) < 2:
            raise ValueError("뱅크에는 클래스가 2개 이상 필요합니다.")
        counts = torch.tensor([(y_train == c).sum() for c in classes])
        r = int(min(self.n_ref, counts.min().item()))
        gen = torch.Generator().manual_seed(seed)
        rows = []
        for c in classes:
            idx = (y_train == c).nonzero(as_tuple=True)[0]
            pick = idx[torch.randperm(len(idx), generator=gen)[:r]]
            rows.append(z_train[pick])
        self.bank = torch.stack(rows)  # (K, R, m)
        self.bank_classes = classes
        k, _, m = self.bank.shape
        flat = self.bank.reshape(k * r, m)
        self.bank_sigma2 = median_sigma2(pairwise_sqdist(flat, flat))
        # μ± 동결: 뱅크 포인트 자신을 쿼리로, 자기 슬롯 제외
        e_pos, e_neg = self._bank_energies(flat, exclude_flat_index=torch.arange(k * r))
        y_bank_local = torch.arange(k, device=flat.device).repeat_interleave(r)
        idx = y_bank_local.view(-1, 1)
        self.bank_mu_pos = e_pos.gather(1, idx).mean().clamp_min(1e-12)
        self.bank_mu_neg = e_neg.gather(1, idx).mean().clamp_min(1e-12)

    @torch.no_grad()
    def _bank_energies(
        self,
        zq: torch.Tensor,
        exclude_flat_index: torch.Tensor | None = None,
        chunk: int = 1024,
    ):
        """쿼리 (N,m) → 뱅크 기준 에너지 (E⁺ (N,K), E⁻ (N,K))."""
        k, r, m = self.bank.shape
        flat = self.bank.reshape(k * r, m)
        out_pos, out_neg = [], []
        for s in range(0, len(zq), chunk):
            q = zq[s : s + chunk]
            c = len(q)
            d2 = pairwise_sqdist(q, flat)  # (C, K*R)
            logw = -d2 / self.bank_sigma2
            if exclude_flat_index is not None:
                excl = exclude_flat_index[s : s + c].to(logw.device)
                logw[torch.arange(c, device=logw.device), excl] = float("-inf")
            logw_v = logw.view(c, k, r)
            d2_v = d2.view(c, k, r)
            # E⁺: 클래스별 soft-min 거리 (softmax가 내부에서 shift 처리)
            e_pos = (torch.softmax(logw_v, dim=-1) * d2_v).sum(-1)  # (C, K)
            # E⁻: 보수 항등식. 행 단위 shift는 비율에서 소거됨
            shift = logw.max(dim=1, keepdim=True).values  # (C, 1)
            w = (logw - shift).exp().double().view(c, k, r)
            s0 = w.sum(-1)  # (C, K)
            s1 = (w * d2_v.double()).sum(-1)
            s0_tot = s0.sum(1, keepdim=True)
            s1_tot = s1.sum(1, keepdim=True)
            e_neg = ((s1_tot - s1) / (s0_tot - s0).clamp_min(1e-300)).to(e_pos.dtype)
            out_pos.append(e_pos)
            out_neg.append(e_neg)
        return torch.cat(out_pos), torch.cat(out_neg)

    @torch.no_grad()
    def forward_bank(self, z: torch.Tensor) -> torch.Tensor:
        """전체 K 클래스 로짓 (N, K) — 동결된 σ²·μ± 사용."""
        assert self.bank is not None, "fit_bank()를 먼저 호출하세요."
        e_pos, e_neg = self._bank_energies(z)
        return -e_pos / self.bank_mu_pos + self.lam * (e_neg / self.bank_mu_neg)

    @torch.no_grad()
    def predict(self, z: torch.Tensor) -> torch.Tensor:
        """원래 클래스 레이블로 예측."""
        return self.bank_classes[self.forward_bank(z).argmax(1)]

    @torch.no_grad()
    def ood_score(self, z: torch.Tensor) -> torch.Tensor:
        """E_total = min_k E⁺_k — 스케일링 전 원시 에너지. 클수록 OOD."""
        e_pos, _ = self._bank_energies(z)
        return e_pos.min(dim=1).values

    # ---------- 뱅크 저장/복원 (npz 연동용) ----------

    def bank_state(self) -> dict:
        return {
            "bank": self.bank.cpu().numpy(),
            "bank_classes": self.bank_classes.cpu().numpy(),
            "bank_sigma2": self.bank_sigma2.cpu().numpy(),
            "bank_mu_pos": self.bank_mu_pos.cpu().numpy(),
            "bank_mu_neg": self.bank_mu_neg.cpu().numpy(),
        }

    def load_bank_state(self, state: dict, device: torch.device | str = "cpu") -> None:
        self.bank = torch.as_tensor(state["bank"]).to(device)
        self.bank_classes = torch.as_tensor(state["bank_classes"]).to(device)
        self.bank_sigma2 = torch.as_tensor(state["bank_sigma2"]).to(device)
        self.bank_mu_pos = torch.as_tensor(state["bank_mu_pos"]).to(device)
        self.bank_mu_neg = torch.as_tensor(state["bank_mu_neg"]).to(device)
