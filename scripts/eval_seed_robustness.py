"""plan12 — 시드 강건성 집계 (재학습 0, results/의 metrics.json·embeddings.npz만 사용).

시드 42(원캐시) + {0,1,2,3}(plan12 신규)의 5시드에 대해 방법별 지표 mean±SD와
시드별 짝지은 델타(부호 일관성 포함)를 집계한다. 프로토콜은 기존 평가 스크립트와 동일:
  - mAP: leave-one-out 코사인 per-query AP (eval_stats_ci.py per_query_ap 규약)
  - few-shot s=1: 20에피소드, seed_base=1000+e (eval_stats_ci.py fewshot_k1_paired 규약)
  - 검증쌍 AUC: balanced 20만+20만 쌍, rng seed 42 (eval_stats_ci.py verif_pair_scores 규약)
  - top1·silhouette·ood_auc: 각 런의 metrics.json
미완료 실험은 자동 스킵(큐 진행 중 중간 점검 가능). 출력: results/seed_robustness.json.

실행: PYTHONUTF8=1 .venv\\Scripts\\python.exe scripts/eval_seed_robustness.py
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 0, 1, 2, 3]  # 42 = 원캐시(접미사 없음), 나머지 = _s<seed>

# 역할 -> 원실험명. CORE는 CIFAR-100(임베딩 지표), OOD는 CIFAR-10->SVHN(ood_auc만).
CORE = {"shrb": "c100v_shrb", "softmax": "cifar100_softmax_lr3", "mhrb": "c100v_mhrb"}
OOD = {"shrb": "ood10v_shrb", "mhrb": "ood10v_mhrb", "softmax": "ood_cifar10_softmax_lr3"}
# Tier3 (있으면 자동 포함)
DML = {"arcface": "cifar100_arcface_lr3", "normsoftmax": "cifar100_normsoftmax_lr3",
       "ms": "cifar100_ms_lr3", "proxyanchor": "cifar100_proxyanchor_plr"}
N_EP = 20


def exp_name(base: str, seed: int) -> str:
    return base if seed == 42 else f"{base}_s{seed}"


def load_metrics(exp: str):
    p = ROOT / "results" / exp / "metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def load_emb(exp: str):
    p = ROOT / "results" / exp / "embeddings.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return z["z_test"].astype(np.float32), z["y_test"].astype(int)


def per_query_ap(z, y, device, chunk=2000):
    """leave-one-out 코사인 검색의 질의별 AP (eval_stats_ci.py와 동일 규약)."""
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


def fewshot_k1(z, y, n_ep=N_EP, seed_base=1000):
    """K=1 NCM, 에피소드 시드 고정(1000+e) — 방법 간·시드 간 동일 에피소드."""
    classes = np.unique(y)
    idx = [np.where(y == c)[0] for c in classes]
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    accs = []
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
        proto = zn[np.array(sup)]
        proto = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-12)
        pred = (zn[qi] @ proto.T).argmax(1)
        accs.append(float((pred == ql).mean()))
    return np.array(accs)


def verif_auc(z, y, n_pairs=200_000, seed=42):
    """balanced 동/이 클래스 쌍 코사인 AUC (eval_stats_ci.py verif_pair_scores 규약)."""
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
    bad = y[i] == y[j]
    while bad.any():
        j[bad] = rng.integers(0, len(y), int(bad.sum()))
        bad = y[i] == y[j]
    negs = (zc[i] * zc[j]).sum(1)
    scores = np.concatenate([pos, negs])
    labels = np.concatenate([np.ones(n_pairs), np.zeros(n_pairs)])
    return float(roc_auc_score(labels, scores))


def stats(vals: dict):
    """{seed: value} -> 요약. 값 2개 미만이면 mean만."""
    v = np.array(list(vals.values()), dtype=float)
    out = {"by_seed": {str(k): float(x) for k, x in vals.items()},
           "n": len(v), "mean": float(v.mean())}
    if len(v) >= 2:
        out["sd"] = float(v.std(ddof=1))
        out["min"], out["max"] = float(v.min()), float(v.max())
    return out


def delta_stats(a: dict, b: dict):
    """짝지은 시드별 델타 a-b (공통 시드만)."""
    common = [s for s in a if s in b]
    d = {s: a[s] - b[s] for s in common}
    if not d:
        return None
    out = stats(d)
    out["n_pos"] = int(sum(1 for x in d.values() if x > 0))
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    core_metrics = {}  # role -> metric -> {seed: val}
    for role, base in {**CORE, **DML}.items():
        acc = {}
        for seed in SEEDS:
            exp = exp_name(base, seed)
            m = load_metrics(exp)
            if m is None:
                continue
            emb = load_emb(exp)
            if emb is None:
                continue
            z, y = emb
            ap = per_query_ap(z, y, device)
            fs = fewshot_k1(z, y)
            acc.setdefault("top1", {})[seed] = m["top1"]
            acc.setdefault("silhouette", {})[seed] = m["silhouette"]
            acc.setdefault("map", {})[seed] = float(ap.mean())
            acc.setdefault("fewshot_s1", {})[seed] = float(fs.mean())
            acc.setdefault("verif_auc", {})[seed] = verif_auc(z, y)
            print(f"[core] {exp}: top1={m['top1']:.4f} mAP={ap.mean():.4f} "
                  f"fs1={fs.mean():.4f} verif={acc['verif_auc'][seed]:.4f}")
        if acc:
            core_metrics[role] = acc

    ood_metrics = {}  # role -> {seed: ood_auc}  (far-OOD: CIFAR-10 -> SVHN)
    for role, base in OOD.items():
        acc = {}
        for seed in SEEDS:
            m = load_metrics(exp_name(base, seed))
            if m is not None and "ood_auc" in m:
                acc[seed] = m["ood_auc"]
                print(f"[ood]  {exp_name(base, seed)}: auc={m['ood_auc']:.4f}")
        if acc:
            ood_metrics[role] = acc

    # near-OOD (CIFAR-10 -> CIFAR-100, eval_near_ood.py 사후 채점 결과가 있으면 포함)
    near_metrics = {}  # role -> {seed: near_auc}
    near_f = ROOT / "results" / "near_ood.json"
    if near_f.exists():
        near_all = json.loads(near_f.read_text(encoding="utf-8"))
        for role, base in OOD.items():
            acc = {s: near_all[exp_name(base, s)]["near_auc"]
                   for s in SEEDS if exp_name(base, s) in near_all}
            if acc:
                near_metrics[role] = acc

    out = {"seeds": SEEDS, "core": {}, "ood": {}, "ood_near": {}, "deltas": {}, "band": {}}
    for role, mm in core_metrics.items():
        out["core"][role] = {k: stats(v) for k, v in mm.items()}
    for role, v in ood_metrics.items():
        out["ood"][role] = stats(v)
    for role, v in near_metrics.items():
        out["ood_near"][role] = stats(v)

    # 짝지은 델타 — plan12 §5-2
    def core_delta(ra, rb, metric):
        if ra in core_metrics and rb in core_metrics and metric in core_metrics[ra]:
            return delta_stats(core_metrics[ra][metric], core_metrics[rb].get(metric, {}))
        return None

    pairs = {
        "shrb_vs_softmax": [("top1", None), ("map", None), ("fewshot_s1", None),
                            ("verif_auc", None), ("silhouette", None)],
        "shrb_vs_mhrb": [("map", None), ("fewshot_s1", None), ("verif_auc", None),
                         ("silhouette", None), ("top1", None)],
    }
    for key, metrics_list in pairs.items():
        ra, rb = ("shrb", "softmax") if key == "shrb_vs_softmax" else ("shrb", "mhrb")
        d = {m: core_delta(ra, rb, m) for m, _ in metrics_list}
        out["deltas"][key] = {k: v for k, v in d.items() if v}
    for tag, mm in (("ood", ood_metrics), ("ood_near", near_metrics)):
        if "shrb" in mm:
            for rb in ("mhrb", "softmax"):
                if rb in mm:
                    d = delta_stats(mm["shrb"], mm[rb])
                    if d:
                        out["deltas"][f"{tag}_shrb_vs_{rb}"] = {"auc": d}

    # 밴드 근거화 (plan12 §5-3): 짝지은 top1 델타의 2*SD vs 0.015
    t1 = out["deltas"].get("shrb_vs_softmax", {}).get("top1")
    if t1 and "sd" in t1:
        out["band"] = {"top1_delta_sd": t1["sd"], "suggested_band_2sd": 2 * t1["sd"],
                       "paper_band": 0.015}

    (ROOT / "results" / "seed_robustness.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    print("\n===== plan12 시드 강건성 요약 =====")
    for role in core_metrics:
        s = out["core"][role]
        line = ", ".join(f"{k} {v['mean']:.4f}" + (f"+-{v['sd']:.4f}" if "sd" in v else "")
                         for k, v in s.items())
        print(f"{role}(n={s['top1']['n']}): {line}")
    for tag in ("ood", "ood_near"):
        for role, s in out[tag].items():
            sd = f"+-{s['sd']:.4f}" if "sd" in s else ""
            print(f"{tag}/{role}(n={s['n']}): auc {s['mean']:.4f}{sd}")
    for key, dd in out["deltas"].items():
        for metric, d in dd.items():
            sd = f"+-{d['sd']:.4f}" if "sd" in d else ""
            print(f"delta {key}/{metric}: {d['mean']:+.4f}{sd} 부호 {d['n_pos']}/{d['n']}")
    if out["band"]:
        b = out["band"]
        print(f"band: 2*SD(top1 delta)={b['suggested_band_2sd']:.4f} vs 고정 밴드 0.015")
    print("saved -> results/seed_robustness.json")


if __name__ == "__main__":
    main()
