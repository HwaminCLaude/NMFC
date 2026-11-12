"""명제 검증 — 명제/정리 후보 8건의 실측 스팟체크.

정리 10건을 우리 실측 형식(비정규화 soft-min LSE + 힌지 + 학습 온도)
에서 검증한다. 재학습 0 — results/c100v_shrb/embeddings.npz의
동결 추론 상태(bank, sigma2, tau, m)만 사용한다.

실행: PYTHONUTF8=1 .venv\\Scripts\\python.exe scripts/audit_theory_claims.py
산출: results/theory_audit.json (이 검증 수치의 단일 출처)
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.nmfc.softmin import hinge_temp_logits, knn_neighbor_sets, softmin_energies  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = "c100v_shrb"
K1, K2, LAM = 3, 32, 0.5
TAU_MIN = 0.01
NQ = 512          # 스팟체크 질의 수
SEED = 42

out: dict = {}
checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    checks.append((name, bool(ok), detail))


def sqdist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.cdist(a, b).pow(2)


def energies(z, bank, y_bank, classes, sigma2, k1=K1, k2=K2):
    d2 = sqdist(z, bank)
    ip, ineg = knn_neighbor_sets(d2, y_bank, classes, k1, k2)
    return softmin_energies(d2, ip, sigma2), softmin_energies(d2, ineg, sigma2), d2, ip, ineg


def main() -> None:
    npz = np.load(os.path.join(ROOT, "results", EXP, "embeddings.npz"))
    z_test = torch.tensor(npz["z_test"], dtype=torch.float64)
    bank = torch.tensor(npz["bank"], dtype=torch.float64)
    y_bank = torch.tensor(npz["bank_labels"])
    classes = torch.tensor(npz["bank_classes"])
    sigma2 = torch.tensor(float(npz["bank_sigma2"]), dtype=torch.float64)
    tau = torch.tensor(float(npz["bank_tau"]), dtype=torch.float64)
    m = torch.tensor(float(npz["bank_m"]), dtype=torch.float64)
    y_test = torch.tensor(npz["y_test"])

    out["config"] = dict(exp=EXP, k1=K1, k2=K2, lam=LAM, tau_min=TAU_MIN,
                         sigma2=float(sigma2), tau=float(tau), hinge_m=float(m),
                         n_bank=int(bank.shape[0]), d=int(bank.shape[1]))

    g = torch.Generator().manual_seed(SEED)
    qi = torch.randperm(z_test.shape[0], generator=g)[:NQ]
    z = z_test[qi]

    # ---------- R (립시츠 정리의 유계 영역) ----------
    r_test = float(z_test.norm(dim=1).max())
    r_bank = float(bank.norm(dim=1).max())
    R = max(r_test, r_bank)
    out["R"] = dict(max_norm_test=r_test, max_norm_bank=r_bank, R=R,
                    mean_norm_test=float(z_test.norm(dim=1).mean()))

    e_pos, e_neg, d2, ip, ineg = energies(z, bank, y_bank, classes, sigma2)

    # ---------- 1. 명제 1: d_min² − σ²ln k₁ ≤ E⁺ ≤ d_min² ----------
    d2_exp = d2.unsqueeze(0).expand(len(classes), len(z), -1)
    d_pos = torch.gather(d2_exp, 2, ip)                       # (P,N,k1)
    dmin2 = d_pos.min(dim=-1).values.transpose(0, 1)          # (N,P)
    lo = dmin2 - sigma2 * np.log(K1)
    up_viol = float((e_pos - dmin2).max())
    lo_viol = float((lo - e_pos).max())
    check("P1 상한 E⁺ ≤ d_min²", up_viol <= 1e-9, f"max(E⁺−d_min²)={up_viol:.3e}")
    check("P1 하한 d_min²−σ²ln k₁ ≤ E⁺", lo_viol <= 1e-9, f"max(하한−E⁺)={lo_viol:.3e}")
    out["prop1_bounds"] = dict(max_upper_violation=up_viol, max_lower_violation=lo_viol,
                               band_width=float(sigma2 * np.log(K1)),
                               mean_gap_to_dmin2=float((dmin2 - e_pos).mean()))

    # ---------- 2. 명제 1: 이웃 고정 하 거리 증가 → E⁺ 증가 ----------
    e_scaled = -sigma2 * torch.logsumexp(-(d_pos * 1.10) / sigma2, dim=-1).transpose(0, 1)
    check("P1 단조성(이웃 고정·거리 ↑ → E⁺ ↑)", bool((e_scaled > e_pos).all()),
          f"min(ΔE⁺)={float((e_scaled - e_pos).min()):.3e}")

    # ---------- 3. 프로토타입 특수화: k₁=1 ⇒ E⁺ = ‖z−p‖² ----------
    e1, _, d2b, ip1, _ = energies(z, bank, y_bank, classes, sigma2, k1=1, k2=K2)
    d1 = torch.gather(d2b.unsqueeze(0).expand(len(classes), len(z), -1), 2, ip1)
    err_proto = float((e1 - d1.squeeze(-1).transpose(0, 1)).abs().max())
    check("프로토타입 특수화(k₁=1 ⇒ E⁺=‖z−p‖²)", err_proto < 1e-9, f"max|Δ|={err_proto:.3e}")
    out["prototype_reduction_maxerr"] = err_proto

    # ---------- 4. 제약 선형 소프트맥스 등가(k₁=1, λ=0) ----------
    p_near = bank[torch.gather(ip1.squeeze(-1).transpose(0, 1), 1,
                               torch.zeros(len(z), len(classes), dtype=torch.long))]
    p_near = bank[ip1.squeeze(-1).transpose(0, 1)]            # (N,P,d) 클래스별 최근접 참조
    g_dist = (-e1) / tau
    g_lin = (2 * torch.einsum("npd,nd->np", p_near, z) - p_near.pow(2).sum(-1)) / tau
    diff = g_dist - g_lin                                      # 클래스 무관 상수(−‖z‖²/τ)여야 함
    spread = float((diff - diff.mean(dim=1, keepdim=True)).abs().max())
    check("선형 소프트맥스 등가(k₁=1·λ=0, 클래스 무관 상수차)", spread < 1e-8,
          f"클래스간 잔차 max={spread:.3e}")
    out["linear_equiv_max_class_spread"] = spread

    # ---------- 5·6. σ→0 / σ→∞ 극한 ----------
    # 수렴률은 명제 1의 밴드폭 σ²ln k₁ 그대로다 — 오차가 σ²에 선형으로 줄어드는지 확인한다.
    conv = []
    for f in (1e-2, 1e-4, 1e-6):
        s_small = torch.tensor(float(sigma2) * f, dtype=torch.float64)
        e_small = -s_small * torch.logsumexp(-d_pos / s_small, dim=-1).transpose(0, 1)
        conv.append((f, float((e_small - dmin2).abs().max()), float(s_small * np.log(K1))))
    rel0 = conv[-1][1] / float(dmin2.abs().mean())
    check("σ→0 극한 → 최근접 제곱거리", rel0 < 1e-6,
          f"σ²×1e-6에서 max|E⁺−d_min²|={conv[-1][1]:.3e} (밴드 {conv[-1][2]:.3e})")

    s_big = torch.tensor(float(sigma2) * 1e4, dtype=torch.float64)
    e_big = -s_big * torch.logsumexp(-d_pos / s_big, dim=-1).transpose(0, 1)
    mean_d2 = d_pos.mean(dim=-1).transpose(0, 1)
    rel_inf = float(((e_big + s_big * np.log(K1) - mean_d2).abs() / mean_d2).max())
    check("σ→∞ 극한 → 이웃 평균 제곱거리(+클래스 공통 상수)", rel_inf < 1e-3,
          f"max 상대오차={rel_inf:.3e}")
    out["sigma_limits"] = dict(rel_err_sigma0=rel0, rel_err_sigma_inf=rel_inf,
                               sigma0_convergence=[dict(factor=f, maxerr=e, band=b) for f, e, b in conv])

    # ---------- 7. 평행이동·직교 불변 ----------
    a = torch.randn(bank.shape[1], generator=g, dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.randn(bank.shape[1], bank.shape[1], generator=g, dtype=torch.float64))
    for name, zt, bt in [("평행이동", z + a, bank + a), ("직교변환", z @ q.T, bank @ q.T)]:
        ep2, en2, *_ = energies(zt, bt, y_bank, classes, sigma2)
        err = max(float((ep2 - e_pos).abs().max()), float((en2 - e_neg).abs().max()))
        check(f"{name} 불변", err < 1e-6, f"max|ΔE|={err:.3e}")
        out.setdefault("invariance_maxerr", {})[name] = err

    # ---------- 8. 등방 스케일: σ² 동반 스케일이면 동차(c²), 동결이면 비불변 ----------
    c = 2.0
    ep_j, en_j, *_ = energies(z * c, bank * c, y_bank, classes, sigma2 * c**2)   # σ² 동반
    err_hom = float((ep_j - c**2 * e_pos).abs().max())
    check("스케일 동차성(σ² 동반 시 E → c²E)", err_hom < 1e-6, f"max|Δ|={err_hom:.3e}")

    ep_s, en_s, *_ = energies(z * c, bank * c, y_bank, classes, sigma2)          # σ² 동결(실추론)
    ratio = float((ep_s / e_pos).median())
    # 힌지 '발동'(클램프) = E⁻ > m 이라 min(E⁻,m)=m 으로 잘리는 비율.
    act0, act1 = float((e_neg > m).double().mean()), float((en_s > m).double().mean())
    check("등방 스케일 비불변(σ²·m 동결 시 c² 배율 아님)", abs(ratio - c**2) > 1e-3,
          f"E⁺ 배율 중앙값={ratio:.3f} ≠ c²={c**2}, 힌지 클램프율 {act0:.3f}→{act1:.3f}")
    out["scale"] = dict(c=c, homogeneity_maxerr=err_hom, frozen_sigma_ratio=ratio,
                        hinge_clamp_rate_before=act0, hinge_clamp_rate_after=act1)

    # ---------- 9. 립시츠 상수 ----------
    zg = z[:128].clone().requires_grad_(True)
    d2g = sqdist(zg, bank)
    ipg, ing = knn_neighbor_sets(d2g, y_bank, classes, K1, K2)
    epg, eng = softmin_energies(d2g, ipg, sigma2), softmin_energies(d2g, ing, sigma2)
    logits = hinge_temp_logits(epg, eng, LAM, m, tau)
    gmax = 0.0
    for k in range(0, len(classes), 10):                       # 클래스 10개마다 표본
        gr = torch.autograd.grad(logits[:, k].sum(), zg, retain_graph=True)[0]
        gmax = max(gmax, float(gr.norm(dim=1).max()))
    bound_tau = 4 * R * (1 + LAM) / float(tau)
    bound_taumin = 4 * R * (1 + LAM) / TAU_MIN
    check("립시츠 상한(실측 경사노름 ≤ 4R(1+λ)/τ)", gmax <= bound_tau,
          f"실측 max‖∇g‖={gmax:.2f} ≤ 상한 {bound_tau:.2f}")
    out["lipschitz"] = dict(measured_max_grad_norm=gmax, bound_with_tau=bound_tau,
                            bound_with_tau_min=bound_taumin, R=R, tau=float(tau))

    # ---------- 10. 마진 분해와 λm 유계 ----------
    yq = y_test[qi]
    cls_pos = torch.searchsorted(classes.contiguous(), yq.contiguous())
    gl = hinge_temp_logits(e_pos, e_neg, LAM, m, tau)
    idx = torch.arange(len(z))
    e_pos_y, e_neg_y = e_pos[idx, cls_pos], e_neg[idx, cls_pos]
    sup_adv = e_pos - e_pos_y[:, None]
    riv_adv = torch.minimum(e_neg_y[:, None], m) - torch.minimum(e_neg, m)
    lhs = gl[idx, cls_pos][:, None] - gl
    rhs = (sup_adv + LAM * riv_adv) / tau
    err_margin = float((lhs - rhs).abs().max())
    check("마진 분해 항등식", err_margin < 1e-8, f"max|Δ|={err_margin:.3e}")

    # 에너지 하한: E ≥ d_min² − σ²ln k ≥ −σ²ln k  ⇒  분리 이득 ≤ m + σ²ln k₂ (유계)
    e_neg_lb = -float(sigma2) * np.log(K2)
    check("음의 에너지 하한 E⁻ ≥ −σ²ln k₂", float(e_neg.min()) >= e_neg_lb - 1e-9,
          f"min E⁻={float(e_neg.min()):.3f} ≥ {e_neg_lb:.3f}")
    riv_max, riv_bound = float(riv_adv.max()), float(m) + float(sigma2) * np.log(K2)
    check("분리 이득 유계 (≤ m + σ²ln k₂)", riv_max <= riv_bound + 1e-9,
          f"max={riv_max:.3f} ≤ 상한 {riv_bound:.3f}")

    # 로짓 상한: g_k ≤ (−E⁺_k + λm)/τ  ⇒  E⁺→∞ 이면 g_k→−∞
    ub = (-e_pos + LAM * m) / tau
    check("로짓 상한 g_k ≤ (−E⁺_k+λm)/τ", bool((gl <= ub + 1e-9).all()),
          f"max(g−상한)={float((gl - ub).max()):.3e}")
    out["margin"] = dict(identity_maxerr=err_margin, max_rival_advantage=riv_max,
                         rival_advantage_bound=riv_bound, hinge_m=float(m),
                         e_neg_lower_bound=e_neg_lb, e_neg_min=float(e_neg.min()))

    # ---------- 보고 ----------
    print(f"{'항목':<44} {'판정':<6} 상세")
    print("-" * 100)
    for name, ok, detail in checks:
        print(f"{name:<44} {'PASS' if ok else 'FAIL':<6} {detail}")
    n_pass = sum(1 for _, ok, _ in checks if ok)
    print("-" * 100)
    print(f"통과 {n_pass}/{len(checks)}   R={R:.3f}  tau={float(tau):.4f}  "
          f"sigma2={float(sigma2):.4f}  m={float(m):.3f}")

    out["checks"] = [dict(name=n, passed=ok, detail=d) for n, ok, d in checks]
    out["n_passed"], out["n_total"] = n_pass, len(checks)
    with open(os.path.join(ROOT, "results", "theory_audit.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("saved: results/theory_audit.json")


if __name__ == "__main__":
    main()
