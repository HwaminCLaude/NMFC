"""작동점 감사 — 명제 2의 '작동점' 실측 + 힌지·경쟁항의 추론시 기여도.

audit_theory_claims.py는 sigma→inf 극한을 sigma^2 x 1e4에서 검증한다(6번 항목). 즉 **작동점
sigma^2=26.66에서는 검증되지 않는다**. 그런데 "전역 중앙값 대역폭에서 작동점이 이웃평균
쪽에 가깝다"는 주장이 mAP 이득의 **기전**으로 쓰인다.
이 스크립트가 그 빈틈을 메운다.

측정 1 — 작동점 위치. f = E+ + sigma^2 ln k1 은 soft-min 이며 f(sigma→0)=d_min^2,
  f(sigma→inf)=mean_k1 d^2 사이를 단조 보간한다. 보간 위치
      pos = (f - d_min^2) / (mean d^2 - d_min^2)  in [0,1]
  pos=0 이면 최근접이웃 레짐, pos=1 이면 이웃평균 레짐. 결정 수준(argmax)까지 확인한다.

측정 2 — 힌지 상한과 경쟁항이 추론 결정에 기여하는가. g = (-E+ + lambda*min(E-, m))/tau 그대로의 전체 로짓과
  lambda=0(양의 에너지만)을 비교하고, 동결 m 하에서의 힌지 클램프율을 전 테스트 칸에서 센다.
  **추론 시점만** 말한다 — 학습 중 임베딩을 형성하는 역할은 이 스크립트의 범위가 아니다.

재학습 0 — results/c100v_shrb/embeddings.npz의 동결 추론 상태만 읽는다.
실행: PYTHONUTF8=1 .venv\\Scripts\\python.exe scripts/audit_operating_regime.py
산출: results/operating_regime_audit.json
"""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.nmfc.softmin import knn_neighbor_sets, softmin_energies  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = "c100v_shrb"
K1, K2, LAM = 3, 32, 0.5
CHUNK = 500

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    checks.append((name, bool(ok), detail))


def pct(v, ps=(5, 50, 95)):
    v = np.asarray(v)
    return {f"p{p}": round(float(np.percentile(v, p)), 6) for p in ps} | {
        "mean": round(float(v.mean()), 6)}


def main() -> None:
    npz = np.load(os.path.join(ROOT, "results", EXP, "embeddings.npz"))
    z_test = torch.tensor(npz["z_test"], dtype=torch.float64)
    bank = torch.tensor(npz["bank"], dtype=torch.float64)
    y_bank = torch.tensor(npz["bank_labels"])
    classes = torch.tensor(npz["bank_classes"])
    sigma2 = torch.tensor(float(npz["bank_sigma2"]), dtype=torch.float64)
    tau = float(npz["bank_tau"])
    m = torch.tensor(float(npz["bank_m"]), dtype=torch.float64)
    y_test = torch.tensor(npz["y_test"]).long()
    P = len(classes)
    ln_k1 = float(np.log(K1))

    out: dict = {"config": dict(exp=EXP, k1=K1, k2=K2, lam=LAM, sigma2=float(sigma2),
                                tau=tau, hinge_m=float(m), n_test=int(z_test.shape[0]),
                                n_bank=int(bank.shape[0]), refs_per_class=int(bank.shape[0] // P),
                                sigma2_ln_k1=float(sigma2) * ln_k1)}

    t_true, rel_true, pos_true, pos_all = [], [], [], []
    sp_pos, sp_neg, margin = [], [], []
    n = 0
    ag_mean = ag_nn = mean_ne_nn = 0
    t1_e = t1_mean = t1_nn = t1_full = t1_lam0 = ag_lam0 = 0
    clamp = cells = 0

    for s in range(0, z_test.shape[0], CHUNK):
        z, yt = z_test[s:s + CHUNK], y_test[s:s + CHUNK]
        d2 = torch.cdist(z, bank).pow(2)
        ip, ineg = knn_neighbor_sets(d2, y_bank, classes, K1, K2)
        e_pos = softmin_energies(d2, ip, sigma2)
        e_neg = softmin_energies(d2, ineg, sigma2)
        d_pos = torch.gather(d2.unsqueeze(0).expand(P, len(z), -1), 2, ip).permute(1, 0, 2)

        dmin2, mean_d2 = d_pos.min(dim=-1).values, d_pos.mean(dim=-1)
        f = e_pos + sigma2 * ln_k1
        pos = (f - dmin2) / (mean_d2 - dmin2).clamp_min(1e-12)
        rel = (f - mean_d2).abs() / mean_d2.clamp_min(1e-12)
        tc = torch.arange(len(z))

        t_true.append((d_pos[tc, yt] / sigma2).flatten().numpy())
        rel_true.append(rel[tc, yt].numpy())
        pos_true.append(pos[tc, yt].numpy())
        pos_all.append(pos.flatten().numpy())

        # --- 결정 수준: 명제 2는 argmax 수준의 주장이다 ---
        pe, pm, pn = e_pos.argmin(1), mean_d2.argmin(1), dmin2.argmin(1)
        ag_mean += int((pe == pm).sum())
        ag_nn += int((pe == pn).sum())
        mean_ne_nn += int((pm != pn).sum())
        t1_e += int((pe == yt).sum())
        t1_mean += int((pm == yt).sum())
        t1_nn += int((pn == yt).sum())

        # --- 힌지·경쟁항 ---
        hinged = torch.minimum(e_neg, m)
        clamp += int((e_neg > m).sum())
        cells += e_neg.numel()
        g_full = (-e_pos + LAM * hinged) / tau
        g_lam0 = (-e_pos) / tau
        pf, pl = g_full.argmax(1), g_lam0.argmax(1)
        t1_full += int((pf == yt).sum())
        t1_lam0 += int((pl == yt).sum())
        ag_lam0 += int((pf == pl).sum())
        sp_pos.append((-e_pos).std(dim=1).numpy())
        sp_neg.append((LAM * hinged).std(dim=1).numpy())
        tk = g_full.topk(2, 1).values
        margin.append((tk[:, 0] - tk[:, 1]).numpy())
        n += len(z)

    cat = np.concatenate
    out["regime"] = dict(
        t_true_class=pct(cat(t_true)),
        rel_err_neighborhood_mean_at_operating_sigma=pct(cat(rel_true)),
        interp_position_true_class=pct(cat(pos_true)),
        interp_position_all_classes=pct(cat(pos_all)),
    )
    out["decision"] = dict(
        n=n,
        top1_argmin_Epos=round(t1_e / n, 4),
        top1_neighborhood_mean_rule=round(t1_mean / n, 4),
        top1_nearest_neighbor_rule=round(t1_nn / n, 4),
        agree_Epos_vs_mean_rule=round(ag_mean / n, 4),
        agree_Epos_vs_nn_rule=round(ag_nn / n, 4),
        mean_rule_differs_from_nn_rule=round(mean_ne_nn / n, 4),
    )
    out["hinge_and_competition"] = dict(
        top1_full_logit=round(t1_full / n, 4),
        top1_lambda0=round(t1_lam0 / n, 4),
        delta_top1_from_competition=round((t1_full - t1_lam0) / n, 4),
        argmax_agreement_full_vs_lambda0=round(ag_lam0 / n, 4),
        argmax_flips=n - ag_lam0,
        hinge_clamp_rate_frozen_m=round(clamp / cells, 8),
        hinge_clamp_cells=clamp,
        total_cells=cells,
        across_class_std_neg_Epos=pct(cat(sp_pos)),
        across_class_std_lambda_hinge=pct(cat(sp_neg)),
        logit_top1_top2_margin=pct(cat(margin)),
    )

    # ---------- Phase I 워밍업 진단(logs → results 단일 출처로 승격) ----------
    # results/ 만 읽어도 되도록, history.json에서 필요한 값을
    # 여기서 계산해 이 JSON에 담는다.
    hpath = os.path.join(ROOT, "logs", EXP, "history.json")
    with open(hpath, encoding="utf-8") as fh:
        hist = json.load(fh)
    tr = int(hist["transition_epoch"])
    ph1 = [x for x in hist["loss"][:tr] if isinstance(x, (int, float))]
    # L = 1/(J+eps)  =>  J = 1/L - eps,  |dL/dJ| = 1/(J+eps)^2 = L^2
    j_ph1 = [1.0 / x - 1e-6 for x in ph1]
    en = hist["embed_norm"]
    fr = hist["fisher_ratio"]
    phi = [x for x in hist["phi"][:tr] if isinstance(x, (int, float))]
    out["warmup"] = dict(
        transition_epoch=tr,
        transition_reason=hist.get("transition_reason"),
        phase1_loss=[round(x, 8) for x in ph1],
        phase1_J=[round(x, 1) for x in j_ph1],
        phase1_J_monotone=all(b >= a for a, b in zip(j_ph1, j_ph1[1:])),
        grad_dL_dJ_first=ph1[0] ** 2,
        grad_dL_dJ_min=min(x ** 2 for x in ph1),
        phi_min=min(phi), phi_max=max(phi),
        embed_norm_start=en[0], embed_norm_warmup_end=en[tr - 1],
        embed_norm_after_switch=en[tr],
        fisher_ratio_start=fr[0], fisher_ratio_warmup_end=fr[tr - 1],
        fisher_ratio_final=fr[-1],
    )
    check("Phase I이 자기 목적함수 J를 단조 개선하는가", out["warmup"]["phase1_J_monotone"],
          f"J={out['warmup']['phase1_J']}  (비단조면 워밍업이 자기 기준조차 못 올림)")
    check("Phase I 경사가 살아 있는가 (|dL/dJ| > 1e-5)", ph1[0] ** 2 > 1e-5,
          f"|dL/dJ| 에폭1={ph1[0]**2:.2e} → 최소 {min(x**2 for x in ph1):.2e} (J^-2 자기감쇠)")
    check("워밍업이 만든 임베딩 스케일이 Phase II에서 유지되는가",
          en[tr] >= en[0],
          f"시작 {en[0]:.2f} → 워밍업끝 {en[tr-1]:.2f} → 전환직후 {en[tr]:.2f} "
          f"(시작값 아래로 내려가면 Phase II가 워밍업 결과를 되돌린 것)")

    p50 = out["regime"]["interp_position_true_class"]["p50"]
    check("작동점이 이웃평균 레짐 (보간위치 중앙값 > 0.9)", p50 > 0.9,
          f"pos 중앙값={p50:.4f} (0=최근접, 1=이웃평균)")
    r95 = out["regime"]["rel_err_neighborhood_mean_at_operating_sigma"]["p95"]
    check("작동 sigma^2에서 이웃평균 근사 상대오차 < 1%", r95 < 0.01, f"p95={r95:.2e}")
    am = out["decision"]["agree_Epos_vs_mean_rule"]
    an = out["decision"]["agree_Epos_vs_nn_rule"]
    check("argmax가 이웃평균 규칙과 일치 (> 최근접 규칙)", am > an,
          f"이웃평균 일치={am:.4f} vs 최근접 일치={an:.4f}, 두 규칙 불일치 칸={out['decision']['mean_rule_differs_from_nn_rule']:.4f}")
    check("힌지 상한이 추론에서 한 번이라도 물리는가", clamp > 0,
          f"클램프 {clamp}/{cells} 칸 = {clamp / cells:.2e}  (0이면 min(E-,m)=E- 항등 → m 무효)")
    d = out["hinge_and_competition"]["delta_top1_from_competition"]
    check("경쟁항이 추론 Top-1을 올리는가", d > 0,
          f"전체 {out['hinge_and_competition']['top1_full_logit']:.4f} vs lambda=0 "
          f"{out['hinge_and_competition']['top1_lambda0']:.4f}  Δ={d:+.4f}, "
          f"argmax 뒤집힘 {out['hinge_and_competition']['argmax_flips']}/{n}")

    w = max(len(c[0]) for c in checks) + 2
    print("항목".ljust(w), "판정".ljust(6), "상세")
    print("-" * 108)
    for name, ok, detail in checks:
        print(name.ljust(w), ("PASS" if ok else "FAIL").ljust(6), detail)
    print("-" * 108)
    npass = sum(1 for _, ok, _ in checks if ok)
    print(f"통과 {npass}/{len(checks)}   sigma2={float(sigma2):.4f} tau={tau:.4f} m={float(m):.4f}")
    print("주의: FAIL은 코드 결함이 아니라 '가정한 기전이 작동점에서 관측되지 않는다'는 뜻이다.")
    out["checks"] = [dict(name=nm, ok=ok, detail=dt) for nm, ok, dt in checks]

    op = os.path.join(ROOT, "results", "operating_regime_audit.json")
    with open(op, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print("saved:", op)


if __name__ == "__main__":
    main()
