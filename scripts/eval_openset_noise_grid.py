"""plan15 §7 — 개방셋 노이즈 그리드 분석 (재학습 0, metrics.json만 사용, 결측 자동 스킵).

마스터 롱테이블(방법×scope×rate×시드) + rate-응답 곡선(시드 평균±SD 밴드) + 격차 곡선
(NMFC−CE vs rate, 소멸점 표기) + DiD 곡선. 캐시 6런(clean·rate5 시드42, plan4/14) 포함.

실행: PYTHONUTF8=1 .venv\\Scripts\\python.exe scripts/eval_openset_noise_grid.py
출력: results/openset_noise_grid/{master.csv, master.json, curves_auc.png, curves_top1.png,
      gap_curve.png, did_curve.png}
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.misc import setup_matplotlib_korean  # noqa: E402

METHODS = {"shrb": "NMFC(shrb)", "softmax": "CE(softmax)"}
SCOPES = ("ns", "na")
RATES = (0, 5, 10, 15, 20)
SEEDS = ("42", "0", "1", "2")
COLS = ["closed_seen_top1", "open_ood_auc", "open_top1_novel", "silhouette"]
NOTE = ("noise_seed=0 실현 1회 공유. 시드 밴드는 학습 시드 {42,0,1,2} 표준편차. "
        "clean·rate5 시드42는 plan4/plan14 캐시 인용.")


def exp_name(meth, scope, rate, seed):
    base = f"c100v_{meth}_open"
    if rate > 0:
        base += f"_{scope}{rate}"
    suf = "" if seed == "42" else f"_s{seed}"
    return base + suf


def load(exp):
    p = ROOT / "results" / exp / "metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    setup_matplotlib_korean()
    out_dir = ROOT / "results" / "openset_noise_grid"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []  # 롱테이블
    for meth in METHODS:
        for scope in SCOPES:
            for rate in RATES:
                if rate == 0 and scope == "na":
                    continue  # clean은 scope 무관 — ns 행에만 기재
                for seed in SEEDS:
                    m = load(exp_name(meth, scope, rate, seed))
                    if m is None:
                        continue
                    row = {"method": meth, "scope": scope if rate else "clean",
                           "rate": rate, "seed": seed, "exp": m["name"]}
                    row.update({c: m.get(c) for c in COLS})
                    ns = m.get("label_noise_stats") or {}
                    row["n_mislabeled"] = (ns.get("n_mislabeled_in_train", 0)
                                           + ns.get("n_contaminated_novel_to_seen", 0)) if ns else 0
                    row["train_size"] = ns.get("train_size", 40000)
                    rows.append(row)
    if not rows:
        print("결과 없음 — 런 완료 후 재실행")
        return

    with open(out_dir / "master.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "master.json").write_text(
        json.dumps({"note": NOTE, "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")

    def grid_vals(meth, scope, rate, col):
        """해당 셀의 시드별 값 배열 (clean은 scope 무관)."""
        sel = [r[col] for r in rows
               if r["method"] == meth and r["rate"] == rate and r[col] is not None
               and (rate == 0 or r["scope"] == scope)]
        return np.array(sel, dtype=float)

    # ---- rate-응답 곡선 (2지표 × 2scope) ----
    for col, fname, ylab in (("open_ood_auc", "curves_auc.png", "open-set OOD AUROC"),
                             ("closed_seen_top1", "curves_top1.png", "closed seen Top-1")):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, scope in zip(axes, SCOPES):
            for meth, marker in (("shrb", "o"), ("softmax", "s")):
                xs, ys, es = [], [], []
                for rate in RATES:
                    v = grid_vals(meth, scope, rate, col)
                    if len(v) == 0:
                        continue
                    xs.append(rate)
                    ys.append(v.mean())
                    es.append(v.std(ddof=1) if len(v) > 1 else 0.0)
                ax.errorbar(xs, ys, yerr=es, marker=marker, capsize=3, label=METHODS[meth])
            ax.set_xlabel("노이즈율 (%)")
            ax.set_xticks(RATES)
            ax.set_title(f"scope={scope} ({'배정 오염' if scope == 'ns' else '경계 오염'})")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel(ylab)
        axes[1].legend()
        fig.suptitle(f"{ylab} — 노이즈율 응답 (오차막대: 시드 SD, n≤4)")
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)

    # ---- 격차 곡선 (NMFC−CE, 시드 짝지음) ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    gap_summary = {}
    for scope, marker in (("ns", "o"), ("na", "s")):
        xs, ys, es, signs = [], [], [], []
        for rate in RATES:
            deltas = []
            for seed in SEEDS:
                a = load(exp_name("shrb", scope, rate, seed))
                b = load(exp_name("softmax", scope, rate, seed))
                if a and b:
                    deltas.append(a["open_ood_auc"] - b["open_ood_auc"])
            if not deltas:
                continue
            d = np.array(deltas)
            xs.append(rate)
            ys.append(d.mean())
            es.append(d.std(ddof=1) if len(d) > 1 else 0.0)
            signs.append(f"{(d > 0).sum()}/{len(d)}")
        ax.errorbar(xs, ys, yerr=es, marker=marker, capsize=3,
                    label=f"scope={scope}")
        gap_summary[scope] = {"rates": xs, "mean": [round(float(y), 4) for y in ys],
                              "sd": [round(float(e), 4) for e in es], "sign_pos": signs}
    ax.axhline(0, color="k", lw=0.8)
    ax.axhspan(-0.015, 0.015, color="gray", alpha=0.15, label="동급 밴드 ±0.015")
    ax.set_xlabel("노이즈율 (%)")
    ax.set_xticks(RATES)
    ax.set_ylabel("open OOD AUC 격차 (NMFC − CE)")
    ax.set_title("개방셋 탐지 격차의 노이즈율 응답 — 소멸점 탐색 (G2)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "gap_curve.png", dpi=150)
    plt.close(fig)

    # ---- DiD 곡선 (Δ_NMFC − Δ_CE vs clean, 서술적) ----
    fig, ax = plt.subplots(figsize=(7, 4))
    for col, ls in (("open_ood_auc", "-"), ("closed_seen_top1", "--")):
        for scope, marker in (("ns", "o"), ("na", "s")):
            xs, ys = [], []
            for rate in RATES[1:]:
                dd = []
                for seed in SEEDS:
                    a0, a1 = load(exp_name("shrb", scope, 0, seed)), load(exp_name("shrb", scope, rate, seed))
                    b0, b1 = load(exp_name("softmax", scope, 0, seed)), load(exp_name("softmax", scope, rate, seed))
                    if all([a0, a1, b0, b1]):
                        dd.append((a1[col] - a0[col]) - (b1[col] - b0[col]))
                if dd:
                    xs.append(rate)
                    ys.append(float(np.mean(dd)))
            if xs:
                ax.plot(xs, ys, marker=marker, ls=ls, label=f"{col}/{scope}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("노이즈율 (%)")
    ax.set_ylabel("DiD = Δ_NMFC − Δ_CE")
    ax.set_title("내성 대결 (음수 = CE가 덜 깎임) — 서술적, 유의 주장 금지")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "did_curve.png", dpi=150)
    plt.close(fig)

    print(json.dumps({"gap_curve": gap_summary, "n_rows": len(rows)}, indent=1, ensure_ascii=False))
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
