"""plan11 — 평가 불확실성 정량화 (재학습 0).

① 질의 단위 부트스트랩 CI: 방법별 per-query AP(10000개) → 1000회 리샘플 → mAP 95% CI
   + 제안법(shrb)−경쟁자 델타 CI (같은 질의 리샘플 = 짝지은 부트스트랩).
② few-shot K=1 20 에피소드(방법 간 동일 에피소드 시드로 짝지음) → mean±SD + 쌍별 차 부호 일관성.
출력: results/stats_ci.json. 실행: PYTHONUTF8=1 .venv\\Scripts\\python.exe scripts/eval_stats_ci.py
"""

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MAIN = "c100v_shrb"  # 제안법
RIVALS = [  # 존재하는 것만 자동 포함 (lr 스윕 완료 후 재실행하면 _lr3·_plr도 잡힘)
    "cifar100_proxyanchor", "cifar100_arcface", "cifar100_ms", "cifar100_normsoftmax",
    "cifar100_softmax", "c100v_mhrb",
    "cifar100_proxyanchor_lr3", "cifar100_proxyanchor_plr", "cifar100_arcface_lr3",
    "cifar100_ms_lr3", "cifar100_normsoftmax_lr3", "cifar100_softmax_lr3",
]
N_BOOT, N_EP, SEED = 1000, 20, 42


def load(exp):
    z = np.load(ROOT / "results" / exp / "embeddings.npz")
    return z["z_test"].astype(np.float32), z["y_test"].astype(int)


def per_query_ap(z, y, device, chunk=2000):
    """leave-one-out 코사인 검색의 질의별 AP 벡터 (eval_retrieval.py와 동일 규약)."""
    z = torch.nn.functional.normalize(torch.as_tensor(z, device=device), dim=1)
    y = torch.as_tensor(y, dtype=torch.long, device=device)
    n = z.shape[0]
    aps = []
    for s in range(0, n, chunk):
        q, yq = z[s:s + chunk], y[s:s + chunk]
        sims = q @ z.t()
        rows = torch.arange(q.shape[0], device=device)
        sims[rows, torch.arange(s, s + q.shape[0], device=device)] = float("-inf")
        order = sims.argsort(1, descending=True)[:, :n - 1]
        rel = (y[order] == yq[:, None]).float()
        rc = rel.sum(1).clamp(min=1)
        prec = rel.cumsum(1) / torch.arange(1, n, device=device).float()
        aps.append(((prec * rel).sum(1) / rc).cpu().numpy())
    return np.concatenate(aps)


def fewshot_k1_paired(zs, y, n_ep=N_EP, seed_base=1000):
    """K=1 NCM 정확도, 방법 간 동일 에피소드(같은 지원 표본)로 짝지음. zs: {name: z}."""
    classes = np.unique(y)
    idx = [np.where(y == c)[0] for c in classes]
    zn = {k: v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12) for k, v in zs.items()}
    accs = {k: [] for k in zs}
    for e in range(n_ep):
        rng = np.random.default_rng(seed_base + e)
        sup, qi, ql = [], [], []
        for ci, ix in enumerate(idx):
            p = ix.copy()
            rng.shuffle(p)
            sup.append(p[0])
            qi.append(p[1:])
            ql.append(np.full(len(p) - 1, ci))
        qi, ql = np.concatenate(qi), np.concatenate(ql)
        for k, z in zn.items():
            proto = z[np.array(sup)]
            proto = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-12)
            pred = (z[qi] @ proto.T).argmax(1)
            accs[k].append(float((pred == ql).mean()))
    return {k: np.array(v) for k, v in accs.items()}


def verif_pair_scores(z, y, n_pairs=200_000, seed=42):
    """공유 balanced 쌍의 코사인 점수 + 라벨 (방법 간 동일 쌍 인덱스로 짝지음)."""
    rng = np.random.default_rng(seed)
    zc = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    classes = np.unique(y)
    idx = [np.where(y == c)[0] for c in classes]
    per = len(idx[0])
    mat = np.stack(idx)
    pc = rng.integers(0, len(classes), n_pairs)
    o1 = rng.integers(0, per, n_pairs)
    o2 = rng.integers(0, per - 1, n_pairs)
    o2 += (o2 >= o1)
    pa, pb = mat[pc, o1], mat[pc, o2]
    pos = (zc[pa] * zc[pb]).sum(1)
    i = rng.integers(0, len(y), n_pairs)
    j = rng.integers(0, len(y), n_pairs)
    # 서로 다른 클래스 보장 위해 필요시 j 재추출
    bad = y[i] == y[j]
    while bad.any():
        j[bad] = rng.integers(0, len(y), int(bad.sum()))
        bad = y[i] == y[j]
    negs = (zc[i] * zc[j]).sum(1)
    scores = np.concatenate([pos, negs])
    labels = np.concatenate([np.ones(n_pairs), np.zeros(n_pairs)])
    return scores, labels


def verification_ci(zs, y, n_boot=200, seed=7):
    """검증쌍 AUC 부트스트랩 CI + shrb 짝지은 델타 CI (공유 쌍)."""
    from sklearn.metrics import roc_auc_score
    sl = {e: verif_pair_scores(zs[e], y) for e in zs}
    labels = next(iter(sl.values()))[1]
    scores = {e: sl[e][0] for e in zs}
    n = len(labels)
    rng = np.random.default_rng(seed)
    bidx = rng.integers(0, n, size=(n_boot, n))
    auc_boot = {e: np.array([roc_auc_score(labels[b], scores[e][b]) for b in bidx]) for e in zs}
    out_auc, out_delta = {}, {}
    for e in zs:
        out_auc[e] = {"auc": float(roc_auc_score(labels, scores[e])),
                      "ci95": [float(np.percentile(auc_boot[e], 2.5)),
                               float(np.percentile(auc_boot[e], 97.5))]}
    for e in zs:
        if e == MAIN:
            continue
        d = auc_boot[MAIN] - auc_boot[e]
        out_delta[e] = {"delta": float(out_auc[MAIN]["auc"] - out_auc[e]["auc"]),
                        "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                        "exceeds_band": bool(abs(out_auc[MAIN]["auc"] - out_auc[e]["auc"]) > 0.015)}
    return out_auc, out_delta


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(SEED)
    have = [e for e in [MAIN] + RIVALS if (ROOT / "results" / e / "embeddings.npz").exists()]
    print(f"[포함] {have}")
    zs, y_ref = {}, None
    for e in have:
        z, y = load(e)
        zs[e] = z
        if y_ref is None:
            y_ref = y
        assert (y == y_ref).all(), f"{e}: y 순서 불일치"

    # ---- ① mAP 부트스트랩 CI ----
    print("\n== mAP 질의 부트스트랩 (1000회, 95% CI) ==")
    ap = {e: per_query_ap(zs[e], y_ref, device) for e in have}
    boot_idx = rng.integers(0, len(y_ref), size=(N_BOOT, len(y_ref)))
    out_map = {}
    for e in have:
        bs = ap[e][boot_idx].mean(1)
        out_map[e] = {"mAP": float(ap[e].mean()), "ci95": [float(np.percentile(bs, 2.5)),
                                                           float(np.percentile(bs, 97.5))]}
        print(f"  {e:28} mAP={out_map[e]['mAP']:.4f}  CI[{out_map[e]['ci95'][0]:.4f}, {out_map[e]['ci95'][1]:.4f}]")
    out_delta = {}
    for e in have:
        if e == MAIN:
            continue
        d = (ap[MAIN] - ap[e])[boot_idx].mean(1)  # 같은 질의 리샘플 = 짝지은 델타
        out_delta[e] = {"delta_mAP": float((ap[MAIN] - ap[e]).mean()),
                        "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                        "significant": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)}
        s = out_delta[e]
        print(f"  Δ(shrb−{e:22}) = {s['delta_mAP']:+.4f}  CI[{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]"
              f"  {'유의' if s['significant'] else '겹침'}")

    # ---- ② few-shot K=1, 20 에피소드 짝지음 ----
    print(f"\n== few-shot K=1 ({N_EP} 에피소드, 짝지은 설계) ==")
    fs = fewshot_k1_paired(zs, y_ref)
    out_fs = {}
    for e in have:
        out_fs[e] = {"mean": float(fs[e].mean()), "sd": float(fs[e].std(ddof=1))}
        print(f"  {e:28} K=1 {out_fs[e]['mean']:.3f} ± {out_fs[e]['sd']:.3f}")
    out_pair = {}
    for e in have:
        if e == MAIN:
            continue
        d = fs[MAIN] - fs[e]
        out_pair[e] = {"mean_delta": float(d.mean()), "sd": float(d.std(ddof=1)),
                       "positive_episodes": int((d > 0).sum()), "n_episodes": N_EP}
        print(f"  Δ(shrb−{e:22}) = {d.mean():+.3f} ± {d.std(ddof=1):.3f}  양수 {int((d > 0).sum())}/{N_EP}")

    # ---- ③ 검증쌍 AUC 부트스트랩 + 짝지은 델타 (±0.015 대역 판정) ----
    print("\n== 검증쌍 AUC 부트스트랩 (공유 쌍, shrb 짝지은 델타) ==")
    v_auc, v_delta = verification_ci(zs, y_ref)
    for e in have:
        a = v_auc[e]
        print(f"  {e:28} AUC={a['auc']:.4f}  CI[{a['ci95'][0]:.4f}, {a['ci95'][1]:.4f}]")
    for e in have:
        if e == MAIN:
            continue
        s = v_delta[e]
        print(f"  Δ(shrb−{e:22}) = {s['delta']:+.4f}  CI[{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]"
              f"  {'대역초과' if s['exceeds_band'] else '대역내(동급)'}")

    (ROOT / "results" / "stats_ci.json").write_text(json.dumps(
        {"map_bootstrap": out_map, "map_delta_vs_shrb": out_delta,
         "fewshot_k1": out_fs, "fewshot_k1_paired_delta": out_pair,
         "verification_auc": v_auc, "verification_delta_vs_shrb": v_delta,
         "n_boot": N_BOOT, "n_episodes": N_EP}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[saved] results/stats_ci.json")


if __name__ == "__main__":
    main()
