"""plan13 — 개방셋(80/20) near-OOD **탐지** AUROC (재학습 0, plan4 산출물 재사용).

plan4의 c100v_shrb_open(NMFC)·c100v_softmax_open(CE) 체크포인트/캐시 임베딩만으로
ID(seen 80 test) vs OOD(novel 20 test) 탐지 AUROC를 5개 점수로 산출한다.

프로토콜(NMFC 추론 그대로): 뱅크=seen 80×64 k-center 코어셋(캐시), σ²=ID(seen)-test 질의로 재추정 후
동결(누수 차단), τ/m/k1/k2=학습 종료값 동결. 동결-train σ²는 검증 게이트 겸 far/near 동일-프로토콜 대비.

실행: PYTHONUTF8=1 .venv\\Scripts\\python.exe scripts/eval_openset_detection.py
출력: results/openset_detection/{scores_*.npz, auroc_table.md, histogram_*.png, config_used.json}
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.image import select_seen_classes  # noqa: E402
from src.models.baselines import LinearHead  # noqa: E402
from src.models.heads_v3 import VariantNMFCHead  # noqa: E402
from src.nmfc.affinity import median_sigma2, pairwise_sqdist  # noqa: E402
from src.nmfc.softmin import hinge_temp_logits  # noqa: E402

from sklearn.metrics import roc_auc_score  # noqa: E402

# ---------------------------------------------------------------- 상수 (전부 체크포인트/config에서)
NMFC_EXP = "c100v_shrb_open"
CE_EXP = "c100v_softmax_open"
N_SEEN, SPLIT_SEED, TOTAL = 80, 0, 100
K1, K2 = 3, 32          # configs/cifar100_openset.yaml shrb override
LAM = 0.5               # base.yaml nmfc.lam
TAU_MIN, M_Q = 0.01, 0.75
N_CORESET, N_REF = 64, 64
BANK_SEED = 42          # fit_coreset_bank seed (체크포인트 학습 시)
N_BOOT, BOOT_SEED = 200, 7   # eval_stats_ci.py 규약과 동일
EQUIV_BAND = 0.015
# far-OOD 참조 (동결-train σ² 프로토콜; 표 참조행)
FAR_NMFC_AUC, FAR_CE_AUC = 0.9699, 0.9772  # ood10v_shrb / ood_cifar10_softmax_lr3
# 검증 게이트 목표 (plan4 캐시)
GATE_NMFC_FROZEN, GATE_CE_MSP100 = 0.8301205625, 0.8049388125

OUT = ROOT / "results" / "openset_detection"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_npz(exp):
    z = np.load(ROOT / "results" / exp / "embeddings.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def bank_energies_full(head, z, chunk=1024):
    """전체 쿼리 → (E⁺ (N,K), E⁻ (N,K)) 코어셋 뱅크 기준 (head._bank_energies 청크 래퍼)."""
    ep, en = [], []
    for s in range(0, len(z), chunk):
        e_pos, e_neg = head._bank_energies(z[s:s + chunk])
        ep.append(e_pos)
        en.append(e_neg)
    return torch.cat(ep), torch.cat(en)


def nmfc_scores(head, z, sigma2):
    """주어진 σ²로 s(min_k E⁺)·s_gap·s_msp(1-max softmax) 계산 (τ/m은 동결값 사용)."""
    head.bank_sigma2 = torch.as_tensor(float(sigma2), device=z.device)
    e_pos, e_neg = bank_energies_full(head, z)             # (N,80)
    s = e_pos.min(dim=1).values                            # min_k E⁺
    e_sorted = e_pos.sort(dim=1).values
    s_gap = e_sorted[:, 0] - e_sorted[:, 1]                # 최근접 − 차상위 (≤0), 클수록 OOD
    logits = hinge_temp_logits(e_pos, e_neg, LAM, head.bank_m, head.bank_tau)
    s_msp = 1.0 - F.softmax(logits, dim=1).max(dim=1).values
    return {k: v.detach().cpu().numpy() for k, v in
            {"s": s, "s_gap": s_gap, "s_msp": s_msp}.items()}


def boot_auc(labels, score, bidx):
    """관측 AUROC + 부트스트랩 표본 벡터 (공유 인덱스 bidx)."""
    auc = float(roc_auc_score(labels, score))
    bs = np.array([roc_auc_score(labels[b], score[b]) for b in bidx])
    return auc, bs


def ci95(bs):
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    print(f"[device] {DEVICE}")
    nmfc = load_npz(NMFC_EXP)
    ce = load_npz(CE_EXP)

    # ---------------- 누수·짝지음·출처 assert ----------------
    seen_ref = select_seen_classes(TOTAL, N_SEEN, SPLIT_SEED)
    assert np.array_equal(np.sort(nmfc["seen_classes"]), seen_ref), "NMFC seen_classes ≠ 재현 분할"
    assert np.array_equal(np.sort(ce["seen_classes"]), seen_ref), "CE seen_classes ≠ 재현 분할"
    y_test = nmfc["y_test"].astype(int)
    assert np.array_equal(y_test, ce["y_test"].astype(int)), "두 모델 y_test 순서 불일치(짝지음 불가)"
    seen_mask = nmfc["seen_mask"].astype(bool)
    assert np.array_equal(seen_mask, ce["seen_mask"].astype(bool)), "seen_mask 불일치"
    assert np.array_equal(seen_mask, np.isin(y_test, seen_ref)), "seen_mask ≠ isin(y, seen)"
    # 테스트 split 출처: 10000 표본 = 100클래스×100 (train은 500/클래스라 이 형태 불가)
    assert len(y_test) == 10000, f"test 표본 {len(y_test)} ≠ 10000"
    cls, cnt = np.unique(y_test, return_counts=True)
    assert len(cls) == TOTAL and set(cnt.tolist()) == {100}, "CIFAR-100 test(100×100) 아님 — split 확인"
    novel_cls = np.unique(y_test[~seen_mask])
    assert len(np.intersect1d(novel_cls, seen_ref)) == 0, "novel 클래스가 seen에 누수"
    assert len(novel_cls) == TOTAL - N_SEEN, f"novel 클래스 수 {len(novel_cls)} ≠ 20"
    # 뱅크: seen 80×64, seen 라벨만
    bank_labels = nmfc["bank_labels"].astype(int)
    assert np.array_equal(np.sort(nmfc["bank_classes"]), seen_ref), "뱅크 클래스 ≠ seen"
    assert set(np.unique(bank_labels).tolist()) <= set(seen_ref.tolist()), "뱅크 라벨이 seen 밖(누수)"
    _, bcnt = np.unique(bank_labels, return_counts=True)
    assert set(bcnt.tolist()) == {N_CORESET}, f"뱅크 클래스당 수 {set(bcnt.tolist())} ≠ {N_CORESET}"
    ood_label = (~seen_mask).astype(int)   # OOD(novel)=1
    n_id, n_ood = int(seen_mask.sum()), int((~seen_mask).sum())
    print(f"[assert 통과] ID={n_id} OOD={n_ood}  seen={N_SEEN} novel={len(novel_cls)}  뱅크={len(bank_labels)}(80×64)")

    # ---------------- NMFC 헤드 복원 + 뱅크 로드 ----------------
    head = VariantNMFCHead(energy="softmin", logit="hinge_temp", lam=LAM, k1=K1, k2=K2,
                           tau_min=TAU_MIN, m_quantile=M_Q, n_coreset=N_CORESET, n_ref=N_REF,
                           detach_pi=False, bank="minibatch").to(DEVICE)
    head.load_bank_state({
        "bank": nmfc["bank"], "bank_labels": nmfc["bank_labels"], "bank_classes": nmfc["bank_classes"],
        "bank_sigma2": nmfc["bank_sigma2"], "bank_tau": nmfc["bank_tau"], "bank_m": nmfc["bank_m"],
        "head_energy": nmfc["head_energy"], "head_logit": nmfc["head_logit"],
    }, device=DEVICE)
    sigma2_frozen = float(nmfc["bank_sigma2"])
    z_nmfc = torch.as_tensor(nmfc["z_test"], dtype=torch.float32, device=DEVICE)

    # σ² 재추정 = ID(seen)-test 질의 × seen 코어셋 뱅크 (OOD 미접촉)
    z_id = z_nmfc[torch.as_tensor(seen_mask, device=DEVICE)]
    sigma2_id = float(median_sigma2(pairwise_sqdist(z_id, head.bank_flat)))
    print(f"[σ²] frozen(train)={sigma2_frozen:.4f}  재추정(ID-test)={sigma2_id:.4f}")

    sc_id = nmfc_scores(head, z_nmfc, sigma2_id)       # 기본(표)
    sc_fz = nmfc_scores(head, z_nmfc, sigma2_frozen)   # 게이트·far/near 대비

    # ---------------- CE 점수 (seen-80 기본, 100 병기) ----------------
    ce_ckpt = torch.load(ROOT / "checkpoints" / CE_EXP / "ckpt.pt", map_location=DEVICE, weights_only=False)
    ce_head = LinearHead(128, TOTAL).to(DEVICE)
    ce_head.load_state_dict(ce_ckpt["head"])
    ce_head.eval()
    with torch.no_grad():
        z_ce = torch.as_tensor(ce["z_test"], dtype=torch.float32, device=DEVICE)
        logit100 = ce_head(z_ce)                       # (N,100)
        col80 = torch.as_tensor(seen_ref, device=DEVICE)
        logit80 = logit100[:, col80]                   # seen 80 로짓
    def msp(lg):
        return (1.0 - F.softmax(lg, dim=1).max(dim=1).values).cpu().numpy()
    def energy(lg):
        return (-torch.logsumexp(lg, dim=1)).cpu().numpy()
    ce_scores = {
        "ce_msp": msp(logit80), "ce_msp_100": msp(logit100),
        "ce_energy": energy(logit80), "ce_energy_100": energy(logit100),
    }

    # ---------------- 검증 게이트 ----------------
    g_nmfc = float(roc_auc_score(ood_label, sc_fz["s"]))
    g_ce = float(roc_auc_score(ood_label, ce_scores["ce_msp_100"]))
    # 캐시 open_scores와 원소 일치도 (재구성 정확성)
    d_nmfc = float(np.max(np.abs(sc_fz["s"] - nmfc["open_scores"])))
    d_ce = float(np.max(np.abs(ce_scores["ce_msp_100"] - ce["open_scores"])))
    print(f"[게이트] NMFC frozen-σ² min_kE⁺ AUROC={g_nmfc:.6f} (목표 {GATE_NMFC_FROZEN:.6f}, |Δscore|max={d_nmfc:.2e})")
    print(f"[게이트] CE 1-MSP(100) AUROC={g_ce:.6f} (목표 {GATE_CE_MSP100:.6f}, |Δscore|max={d_ce:.2e})")
    gate_ok = (abs(g_nmfc - GATE_NMFC_FROZEN) < 1e-3 and abs(g_ce - GATE_CE_MSP100) < 1e-3)
    assert gate_ok, "검증 게이트 실패 — 재구성(부호/뱅크/k) 확인 후 중단"
    print("[게이트] 통과 ✓ — 신규 점수 계산 진행")

    # ---------------- 점수 집계 (표 기본 = ID-test σ²) ----------------
    all_scores = {
        "s": sc_id["s"], "s_gap": sc_id["s_gap"], "s_msp": sc_id["s_msp"],
        "s_frozen": sc_fz["s"],
        "ce_msp": ce_scores["ce_msp"], "ce_msp_100": ce_scores["ce_msp_100"],
        "ce_energy": ce_scores["ce_energy"], "ce_energy_100": ce_scores["ce_energy_100"],
    }
    labels_disp = {
        "s": "NMFC  s = min_k E⁺  (ID-test σ²)",
        "s_gap": "NMFC  s_gap = 최근접−차상위 E⁺",
        "s_msp": "NMFC  s_msp = 1−max softmax(로짓)",
        "s_frozen": "NMFC  s = min_k E⁺  (frozen-train σ²)",
        "ce_msp": "CE    1−MSP  (seen-80 로짓)",
        "ce_msp_100": "CE    1−MSP  (100 로짓)",
        "ce_energy": "CE    Energy = −logΣexp  (seen-80)",
        "ce_energy_100": "CE    Energy = −logΣexp  (100)",
    }

    # ---------------- 부트스트랩 (공유 인덱스) ----------------
    rng = np.random.default_rng(BOOT_SEED)
    bidx = rng.integers(0, len(y_test), size=(N_BOOT, len(y_test)))
    auc, boot = {}, {}
    for k, sc in all_scores.items():
        auc[k], boot[k] = boot_auc(ood_label, sc, bidx)
        c = ci95(boot[k])
        print(f"  {labels_disp[k]:44} AUROC={auc[k]:.4f}  CI[{c[0]:.4f},{c[1]:.4f}]")

    # 짝지은 Δ: 제안 s(ID-test σ²) − CE 기준선들 + 다른 NMFC 점수 − ce_msp
    pairs = [("s", "ce_msp"), ("s", "ce_energy"), ("s", "ce_msp_100"),
             ("s_gap", "ce_msp"), ("s_msp", "ce_msp"), ("s_frozen", "ce_msp")]
    deltas = {}
    for a, b in pairs:
        d = boot[a] - boot[b]
        delta = auc[a] - auc[b]
        deltas[f"{a}__vs__{b}"] = {
            "delta": float(delta), "ci95": ci95(d),
            "exceeds_band": bool(abs(delta) > EQUIV_BAND),
            "verdict": "우위" if delta > EQUIV_BAND else ("열위" if delta < -EQUIV_BAND else "동급(±0.015)"),
        }
        v = deltas[f"{a}__vs__{b}"]
        print(f"  Δ({a} − {b}) = {delta:+.4f}  CI[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]  {v['verdict']}")

    # far/near 하락 폭 (동일 프로토콜: NMFC는 frozen-σ² near, CE는 seen-80 near)
    drop_nmfc = FAR_NMFC_AUC - auc["s_frozen"]
    drop_ce = FAR_CE_AUC - auc["ce_msp"]
    print(f"[far→near 하락] NMFC(frozen σ²) {FAR_NMFC_AUC:.4f}→{auc['s_frozen']:.4f} (−{drop_nmfc:.4f}) | "
          f"CE 1-MSP {FAR_CE_AUC:.4f}→{auc['ce_msp']:.4f} (−{drop_ce:.4f})")

    # ---------------- 산출물 저장 ----------------
    for k, sc in all_scores.items():
        np.savez(OUT / f"scores_{k}.npz", score=sc.astype(np.float32),
                 ood_flag=ood_label.astype(np.int8), y_true=y_test.astype(np.int64),
                 is_id=seen_mask.astype(np.int8))

    write_table(auc, boot, deltas, labels_disp, drop_nmfc, drop_ce)
    make_histograms(all_scores, seen_mask, auc, labels_disp)
    save_config(sigma2_frozen, sigma2_id, seen_ref, auc, deltas, drop_nmfc, drop_ce,
                g_nmfc, g_ce, n_id, n_ood)
    print(f"\n[완료] 산출물 → {OUT}")


def write_table(auc, boot, deltas, labels_disp, drop_nmfc, drop_ce):
    L = ["# 개방셋(80/20) near-OOD 탐지 AUROC — plan13", "",
         "ID = seen 80클래스 test(8000), OOD = novel 20클래스 test(2000). AUROC: ID=0, OOD=1.",
         f"부트스트랩 {N_BOOT}회 95% CI(percentile), 짝지은 Δ는 공유 리샘플 인덱스. 동급 밴드 |Δ|≤{EQUIV_BAND}.", "",
         "## 방법 × 점수 AUROC", "", "| 점수 | AUROC | 95% CI |", "|---|---|---|"]
    order = ["s", "s_gap", "s_msp", "s_frozen", "ce_msp", "ce_energy", "ce_msp_100", "ce_energy_100"]
    for k in order:
        c = ci95(boot[k])
        L.append(f"| {labels_disp[k]} | {auc[k]:.4f} | [{c[0]:.4f}, {c[1]:.4f}] |")
    L += ["", "## 짝지은 Δ (제안 − 기준선) + 동급 판정", "", "| 대비 | Δ AUROC | 95% CI | 판정 |", "|---|---|---|---|"]
    for key, v in deltas.items():
        a, b = key.split("__vs__")
        L.append(f"| {a} − {b} | {v['delta']:+.4f} | [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] | {v['verdict']} |")
    L += ["", "## far-OOD 참조 (CIFAR-10→SVHN) 와 far→near 하락 폭", "",
          "| 프로토콜 | far-OOD | near-OOD | 하락 |", "|---|---|---|---|",
          f"| NMFC min_k E⁺ (frozen-train σ², 동일 프로토콜) | {FAR_NMFC_AUC:.4f} | {auc['s_frozen']:.4f} | −{drop_nmfc:.4f} |",
          f"| CE 1−MSP | {FAR_CE_AUC:.4f} | {auc['ce_msp']:.4f} | −{drop_ce:.4f} |", "",
          "> ⚠️ far-OOD 참조행은 **별개 모델**의 기존 수치다(제안 `ood10v_shrb`, CE `ood_cifar10_softmax_lr3` — "
          "둘 다 CIFAR-10→SVHN 학습, frozen-train σ²). 이 표의 near-OOD(seen80 CIFAR-100 모델)와 **학습 데이터·"
          "OOD 셋·로짓 카디널리티가 모두 달라**, far→near '하락'은 동일 모델의 열화가 아니라 **교란된 맥락 대비**다. "
          "확신도(CE)가 더 크게 하락한 것은 가설과 일치하나, 동일 seen80 모델의 far-OOD가 없어 이를 청정 증거로 "
          "삼지 않는다(적대 감사 지적 반영). 청정 증거는 위 짝지은 Δ(동일 모델·동일 근-OOD 과제)다."]
    (OUT / "auroc_table.md").write_text("\n".join(L), encoding="utf-8")


def make_histograms(all_scores, seen_mask, auc, labels_disp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.utils import setup_matplotlib_korean
    setup_matplotlib_korean()
    for k, sc in all_scores.items():
        idv, oodv = sc[seen_mask], sc[~seen_mask]
        lo, hi = np.percentile(sc, [0.5, 99.5])   # 이상치 클리핑(−J/음수 점수 대비)
        if hi <= lo:
            hi = lo + 1e-6
        bins = np.linspace(lo, hi, 60)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(np.clip(idv, lo, hi), bins=bins, density=True, alpha=0.55, label=f"ID (seen, n={len(idv)})", color="#1f77b4")
        ax.hist(np.clip(oodv, lo, hi), bins=bins, density=True, alpha=0.55, label=f"OOD (novel, n={len(oodv)})", color="#d62728")
        safe = labels_disp[k].replace("⁺", "+").replace("−", "-")  # Malgun Gothic 결손 글리프
        ax.set_title(f"{safe}\nAUROC={auc[k]:.4f}", fontsize=10)
        ax.set_xlabel("점수 (클수록 OOD)")
        ax.set_ylabel("밀도")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / f"histogram_{k}.png", dpi=130)
        plt.close(fig)


def save_config(sig_fz, sig_id, seen_ref, auc, deltas, drop_nmfc, drop_ce, g_nmfc, g_ce, n_id, n_ood):
    cfg = {
        "plan": "plan13", "task": "openset 80/20 near-OOD detection AUROC (재학습 0)",
        "checkpoints": {
            "nmfc": f"checkpoints/{NMFC_EXP}/ckpt.pt", "ce": f"checkpoints/{CE_EXP}/ckpt.pt",
            "nmfc_embeddings": f"results/{NMFC_EXP}/embeddings.npz",
            "ce_embeddings": f"results/{CE_EXP}/embeddings.npz",
        },
        "split": {"n_seen": N_SEEN, "split_seed": SPLIT_SEED, "total": TOTAL,
                  "seen_classes": seen_ref.tolist(), "n_id_test": n_id, "n_ood_test": n_ood},
        "nmfc_hparams": {"k1": K1, "k2": K2, "lam": LAM, "tau_min": TAU_MIN, "m_quantile": M_Q,
                         "n_coreset": N_CORESET, "energy": "softmin", "logit": "hinge_temp",
                         "bank_tau_frozen": float(np.load(ROOT / "results" / NMFC_EXP / "embeddings.npz")["bank_tau"]),
                         "bank_m_frozen": float(np.load(ROOT / "results" / NMFC_EXP / "embeddings.npz")["bank_m"])},
        "sigma2": {"frozen_train": sig_fz, "reestimated_id_test": sig_id,
                   "primary_for_table": "reestimated_id_test",
                   "note": "σ² 재추정 질의는 ID(seen)-test만; OOD·뱅크 미접촉. τ·m 동결이라 σ² 변화는 s_msp에 주로 영향."},
        "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED, "ci": "percentile [2.5,97.5]", "equiv_band": EQUIV_BAND},
        "gate": {"nmfc_frozen_target": GATE_NMFC_FROZEN, "nmfc_frozen_measured": g_nmfc,
                 "ce_msp100_target": GATE_CE_MSP100, "ce_msp100_measured": g_ce, "passed": True},
        "auroc": {k: float(v) for k, v in auc.items()},
        "paired_delta": deltas,
        "far_ood_reference": {"nmfc_ood10v_shrb": FAR_NMFC_AUC, "ce_ood_cifar10_softmax_lr3": FAR_CE_AUC,
                              "drop_nmfc_frozen": float(drop_nmfc), "drop_ce_msp": float(drop_ce)},
        "seed": BANK_SEED,
    }
    (OUT / "config_used.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
