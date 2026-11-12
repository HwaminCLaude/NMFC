"""v3 변형 헤드 — soft-min/mean 에너지 × 힌지·온도 로짓 × 배치/온라인 뱅크.

v1 NMFCHead(mean+mean_scaled)는 그대로 두고(is_v1 조합), 비-v1 조합만 이 헤드가 담당한다.
로짓은 항상 hinge_temp이며(mean_scaled는 v1 전용), 에너지는 mean(v1 정규화 π 재사용) 또는
softmin(v3 log-sum-exp) 중 하나다. 학습 참조는 배치 내(minibatch) 또는 온라인 뱅크,
추론 참조는 동결 k-center 코어셋이다. σ²·τ·m은 추론 시 동결한다.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.banks import kcenter_greedy
from src.nmfc.affinity import median_sigma2, pairwise_sqdist, soft_neighbor_weights
from src.nmfc.energy import local_energies
from src.nmfc.softmin import hinge_temp_logits, knn_neighbor_sets, softmin_energies


class VariantNMFCHead(nn.Module):
    def __init__(
        self,
        energy: str,
        logit: str,
        lam: float,
        k1: int,
        k2: int,
        tau_min: float,
        m_quantile: float,
        n_coreset: int,
        n_ref: int,
        detach_pi: bool,
        bank: str = "minibatch",
    ):
        super().__init__()
        assert logit == "hinge_temp", "VariantNMFCHead는 hinge_temp 로짓 전용 (mean_scaled는 v1 NMFCHead)"
        self.energy = energy
        self.logit = logit
        self.bank_mode = bank
        self.lam = lam
        self.k1 = k1
        self.k2 = k2
        self.tau_min = tau_min
        self.m_quantile = m_quantile
        self.n_coreset = n_coreset
        self.n_ref = n_ref
        self.detach_pi = detach_pi
        # 학습 온도 τ = τ_min + softplus(ρ), ρ₀은 τ₀=1.0이 되게
        rho0 = math.log(math.exp(1.0 - tau_min) - 1.0)
        self.rho = nn.Parameter(torch.tensor(float(rho0)))
        self.register_buffer("m", torch.tensor(float("nan")))  # t_c에 set_margin으로 앵커
        # 추론 코어셋 뱅크 (fit_coreset_bank에서 채움)
        self.bank_flat: torch.Tensor | None = None
        self.bank_labels: torch.Tensor | None = None
        self.bank_classes: torch.Tensor | None = None
        self.register_buffer("bank_sigma2", torch.tensor(float("nan")))
        self.register_buffer("bank_tau", torch.tensor(float("nan")))
        self.register_buffer("bank_m", torch.tensor(float("nan")))

    @property
    def tau(self) -> torch.Tensor:
        return self.tau_min + F.softplus(self.rho)

    # ---------- 에너지 (배치/온라인 공통 하위 계산) ----------

    def _batch_energies(self, z, y, classes):
        """배치 내 참조 E±(N,P). softmin이면 kNN+LSE(자기 LOO), mean이면 v1 정규화 π."""
        d2 = pairwise_sqdist(z, z)
        sigma2 = median_sigma2(d2)  # 정방 → 대각 제외
        if self.energy == "softmin":
            exclude = torch.arange(len(z), device=z.device)
            idx_pos, idx_neg = knn_neighbor_sets(d2, y, classes, self.k1, self.k2, exclude_cols=exclude)
            return softmin_energies(d2, idx_pos, sigma2), softmin_energies(d2, idx_neg, sigma2)
        pi_pos, pi_neg = soft_neighbor_weights(d2, y, classes, sigma2, exclude_self=True)
        if self.detach_pi:
            pi_pos, pi_neg = pi_pos.detach(), pi_neg.detach()
        return local_energies(d2, pi_pos, pi_neg)

    # ---------- 학습 경로 ----------

    def forward_batch(self, z: torch.Tensor, y: torch.Tensor):
        """배치 내 참조로 로짓 (N,P)·로컬 레이블. sampled softmax(배치 등장 클래스)."""
        classes, y_local = torch.unique(y, return_inverse=True)
        if len(classes) < 2:
            raise ValueError("NMFC 배치에는 클래스가 2개 이상 필요합니다 (PK 샘플러).")
        e_pos, e_neg = self._batch_energies(z, y, classes)
        g = hinge_temp_logits(e_pos, e_neg, self.lam, self.m, self.tau)
        return g, y_local

    def forward_online(self, z: torch.Tensor, y: torch.Tensor, bank, idx: torch.Tensor):
        """온라인 뱅크 참조로 전체 K 클래스 로짓 (N,K). LOO는 idx 태그 일치 열 제외."""
        z_ref, y_ref, idx_ref = bank.reference()
        classes = torch.arange(bank.n_classes, device=z.device)
        d2 = pairwise_sqdist(z, z_ref)  # (N, T)
        sigma2 = median_sigma2(d2)  # 사각 → 전체 중앙값
        exclude = self._loo_cols(idx.to(z.device), idx_ref)
        idx_pos, idx_neg = knn_neighbor_sets(d2, y_ref, classes, self.k1, self.k2, exclude_cols=exclude)
        e_pos = softmin_energies(d2, idx_pos, sigma2)  # (N, K)
        e_neg = softmin_energies(d2, idx_neg, sigma2)
        g = hinge_temp_logits(e_pos, e_neg, self.lam, self.m, self.tau)
        return g, y

    @staticmethod
    def _loo_cols(query_idx: torch.Tensor, idx_ref: torch.Tensor) -> torch.Tensor:
        """각 쿼리 sample-index에 대해 idx_ref 일치 열 인덱스((N,), 없으면 −1)."""
        eq = query_idx[:, None] == idx_ref[None, :]  # (N, T)
        has = eq.any(dim=1)
        first = eq.float().argmax(dim=1)  # 첫 일치 열
        return torch.where(has, first, torch.full_like(query_idx, -1))

    @torch.no_grad()
    def margin_energies(self, z, y, bank=None, idx=None) -> torch.Tensor:
        """정답 클래스 척력 에너지 E⁻_{y_i} (N,) — t_c 힌지 마진 앵커용."""
        classes, y_local = torch.unique(y, return_inverse=True)
        if bank is not None:  # online: 뱅크 참조
            z_ref, y_ref, idx_ref = bank.reference()
            d2 = pairwise_sqdist(z, z_ref)
            sigma2 = median_sigma2(d2)
            exclude = self._loo_cols(idx.to(z.device), idx_ref) if idx is not None else None
            _, idx_neg = knn_neighbor_sets(d2, y_ref, classes, self.k1, self.k2, exclude_cols=exclude)
            e_neg = softmin_energies(d2, idx_neg, sigma2)
        else:  # minibatch: 자기 집합 참조(LOO)
            _, e_neg = self._batch_energies(z, y, classes)
        return e_neg.gather(1, y_local.view(-1, 1)).squeeze(1)

    def set_margin(self, e_neg_true: torch.Tensor) -> None:
        """힌지 마진 m = quantile(E⁻_{y_i}, m_quantile). t_c에 1회."""
        self.m = torch.quantile(e_neg_true.detach().float(), self.m_quantile).to(self.m.device)

    # ---------- 추론 경로 (코어셋 뱅크) ----------

    @torch.no_grad()
    def fit_coreset_bank(self, z_train: torch.Tensor, y_train: torch.Tensor, seed: int = 42) -> None:
        """클래스별 k-center 코어셋으로 뱅크 구성, σ²·τ·m 동결."""
        assert self.n_coreset >= self.k1, f"n_coreset={self.n_coreset} < k1={self.k1}"
        classes = torch.unique(y_train)
        rows, labels = [], []
        for c in classes:
            zc = z_train[y_train == c]
            sel = kcenter_greedy(zc, self.n_coreset, seed=seed)
            rows.append(zc[sel])
            labels.append(torch.full((len(sel),), int(c), dtype=torch.long, device=zc.device))
        self.bank_flat = torch.cat(rows)
        self.bank_labels = torch.cat(labels)
        self.bank_classes = classes.to(self.bank_flat.device)
        # σ² 동결: 학습셋 쿼리 × 코어셋 거리 중앙값(질의 분포 대리)
        n = len(z_train)
        if n > 4096:
            g = torch.Generator().manual_seed(seed)
            q = z_train[torch.randperm(n, generator=g)[:4096]]
        else:
            q = z_train
        self.bank_sigma2 = median_sigma2(pairwise_sqdist(q, self.bank_flat))
        self.bank_tau = self.tau.detach()
        self.bank_m = self.m.detach().clone()

    @torch.no_grad()
    def _bank_energies(self, zq: torch.Tensor):
        """쿼리 (C,dim) → (E⁺ (C,K), E⁻ (C,K)) 코어셋 뱅크 기준. energy 타입 분기."""
        d2 = pairwise_sqdist(zq, self.bank_flat)  # (C, T)
        if self.energy == "softmin":
            idx_pos, idx_neg = knn_neighbor_sets(
                d2, self.bank_labels, self.bank_classes, self.k1, self.k2
            )
            return (
                softmin_energies(d2, idx_pos, self.bank_sigma2),
                softmin_energies(d2, idx_neg, self.bank_sigma2),
            )
        pi_pos, pi_neg = soft_neighbor_weights(
            d2, self.bank_labels, self.bank_classes, self.bank_sigma2, exclude_self=False
        )
        return local_energies(d2, pi_pos, pi_neg)

    @torch.no_grad()
    def forward_bank(self, z: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
        """전체 K 클래스 로짓 (N,K) — 동결 σ²·τ·m 사용."""
        assert self.bank_flat is not None, "fit_coreset_bank()를 먼저 호출하세요."
        outs = []
        for s in range(0, len(z), chunk):
            e_pos, e_neg = self._bank_energies(z[s : s + chunk])
            outs.append(hinge_temp_logits(e_pos, e_neg, self.lam, self.bank_m, self.bank_tau))
        return torch.cat(outs)

    @torch.no_grad()
    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return self.bank_classes[self.forward_bank(z).argmax(1)]

    @torch.no_grad()
    def ood_score(self, z: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
        """이상 점수 s(x) = min_k E⁺_k (명제 1). 클수록 OOD."""
        outs = []
        for s in range(0, len(z), chunk):
            e_pos, _ = self._bank_energies(z[s : s + chunk])
            outs.append(e_pos.min(dim=1).values)
        return torch.cat(outs)

    # ---------- 저장/복원 (npz 연동) ----------

    def bank_state(self) -> dict:
        return {
            "bank": self.bank_flat.cpu().numpy(),
            "bank_labels": self.bank_labels.cpu().numpy(),
            "bank_classes": self.bank_classes.cpu().numpy(),
            "bank_sigma2": self.bank_sigma2.cpu().numpy(),
            "bank_tau": self.bank_tau.cpu().numpy(),
            "bank_m": self.bank_m.cpu().numpy(),
            "head_energy": np.array(self.energy),
            "head_logit": np.array(self.logit),
        }

    def load_bank_state(self, state: dict, device="cpu") -> None:
        self.bank_flat = torch.as_tensor(state["bank"]).to(device)
        self.bank_labels = torch.as_tensor(state["bank_labels"]).to(device)
        self.bank_classes = torch.as_tensor(state["bank_classes"]).to(device)
        self.bank_sigma2 = torch.as_tensor(state["bank_sigma2"]).to(device)
        self.bank_tau = torch.as_tensor(state["bank_tau"]).to(device)
        self.bank_m = torch.as_tensor(state["bank_m"]).to(device)
        self.energy = str(state["head_energy"])
        self.logit = str(state["head_logit"])
