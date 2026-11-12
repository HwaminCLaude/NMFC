"""plan7 — 임베딩 검색·군집 평가 (재학습 0, 캐시 embeddings.npz만 사용).

헤드를 제거하고 임베딩 거리만으로 shrb(에너지 기반) vs softmax를 겨룬다.
top-1(softmax의 학습된 헤드 덕)이 아니라 검색(Recall@1/5, mAP)·군집(purity, NMI)으로
"실루엣 우위가 실용 성능으로 환금되는가"를 시험. src/ 불변 — 순수 분석 스크립트.

실행: PYTHONUTF8=1 .venv\\Scripts\\python.exe scripts/eval_retrieval.py
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]

# (그룹 라벨, 실험명) — 같은 트랙끼리 나란히. 전부 224px / embed_dim 128 / 100클래스.
GROUPS = [
    ("A_pretrained40_core", ["c100v_shrb", "cifar100_softmax"]),
    ("B_scratch200_core", ["c100s_shrb", "c100s_softmax"]),
    ("baselines_pretrained40", [
        "cifar100_nmfc", "c100v_mhrb", "cifar100_arcface",
        "cifar100_proto", "cifar100_deepmfa", "cifar100_triplet",
    ]),
    ("D_strong_dml", [  # plan9 — 최신 강한 DML 베이스라인 (없으면 스킵)
        "cifar100_normsoftmax", "cifar100_ms", "cifar100_proxyanchor",
    ]),
    ("E_variant_ablation", [  # v3·v4 변형 모듈 절제(−J·온라인·구면/포화)의 검색축 보강
        "c100v_shtb", "c100v_shro", "c100v_shjb",
        "c100v_shnb", "c100v_shjo", "c100v_shno",
    ]),
    ("C_external_retrieval", [  # plan8 — 외부/오프더셸프 검색모델 (없으면 스킵)
        "ext_resnet50_frozen", "ext_vit_b16", "ext_convnext_base",
        "ext_clip_vitb32", "ext_dinov2_vits14",
    ]),
    ("B2_scratch_dml", [  # plan10 — DML 베이스라인 from-scratch (대조: c100s_shrb/softmax)
        "c100s_arcface", "c100s_normsoftmax", "c100s_ms",
        "c100s_proxyanchor", "c100s_proto", "c100s_deepmfa",
    ]),
    ("D2_dml_lr_sweep", [  # plan11 — 비교군을 제안법 레시피(lr3e-4·48×4)로 재학습 (없으면 스킵)
        "cifar100_arcface_lr3", "cifar100_normsoftmax_lr3", "cifar100_ms_lr3",
        "cifar100_proxyanchor_lr3", "cifar100_proxyanchor_plr", "cifar100_softmax_lr3",
    ]),
]


def average_precision(rel):
    """rel: (Q, M) 0/1 텐서, 각 행은 랭크순 relevance. 행별 AP 반환."""
    r_count = rel.sum(1).clamp(min=1)
    cum = rel.cumsum(1)
    ranks = torch.arange(1, rel.shape[1] + 1, device=rel.device).float()
    prec_at_k = cum / ranks
    return (prec_at_k * rel).sum(1) / r_count


def retrieval_metrics(z, y, device, chunk=2000):
    """leave-one-out 코사인 검색. Recall@1, Recall@5, mAP + 질문당 동클래스 수(R) 반환."""
    z = torch.nn.functional.normalize(
        torch.as_tensor(z, dtype=torch.float32, device=device), dim=1)
    y = torch.as_tensor(y, dtype=torch.long, device=device)
    n = z.shape[0]
    r1 = r5 = ap_sum = 0.0
    r_per_query = None
    for s in range(0, n, chunk):
        q = z[s:min(s + chunk, n)]
        yq = y[s:min(s + chunk, n)]
        c = q.shape[0]
        sims = q @ z.t()  # (c, n)
        rows = torch.arange(c, device=device)
        cols = torch.arange(s, s + c, device=device)
        sims[rows, cols] = float("-inf")  # 자기 자신 제외
        order = sims.argsort(1, descending=True)[:, :n - 1]  # 자기(-inf)가 맨 끝 → 절단
        rel = (y[order] == yq[:, None]).float()
        r1 += rel[:, 0].sum().item()
        r5 += (rel[:, :5].sum(1) > 0).float().sum().item()
        ap_sum += average_precision(rel).sum().item()
        if r_per_query is None:
            r_per_query = int(rel.sum(1)[0].item())
    return {"recall@1": r1 / n, "recall@5": r5 / n, "mAP": ap_sum / n, "R": r_per_query}


def cluster_metrics(z, y, k=100, seed=42):
    """KMeans(코사인) 군집 순도 + NMI."""
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    labels = KMeans(n_clusters=k, random_state=seed, n_init=4).fit(zn).labels_
    hit = sum(np.bincount(y[labels == cl]).max() for cl in np.unique(labels))
    return {"purity": float(hit / len(y)),
            "nmi": float(normalized_mutual_info_score(y, labels))}


def selftest():
    rel = torch.tensor([[1.0, 0, 1], [0, 1, 1]])
    ap = average_precision(rel)
    assert abs(ap[0].item() - 0.833333) < 1e-4, ap[0].item()
    assert abs(ap[1].item() - 0.583333) < 1e-4, ap[1].item()
    y, cl = np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1])
    hit = sum(np.bincount(y[cl == c]).max() for c in np.unique(cl))
    assert hit / 4 == 0.5
    print("[selftest] AP/purity OK")


def load(exp):
    z = np.load(ROOT / "results" / exp / "embeddings.npz")
    m = json.loads((ROOT / "results" / exp / "metrics.json").read_text(encoding="utf-8"))
    return z["z_test"], z["y_test"].astype(int), m


def main():
    selftest()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    rows = []
    for group, exps in GROUPS:
        for exp in exps:
            if not (ROOT / "results" / exp / "embeddings.npz").exists():
                print(f"  skip {exp} (embeddings 없음)")
                continue
            z, y, m = load(exp)
            r = retrieval_metrics(z, y, device)
            assert r["R"] == 99, f"{exp}: 질문당 동클래스 {r['R']} != 99 (누수/정렬 오류)"
            cl = cluster_metrics(z, y)
            rows.append({"group": group, "exp": exp, "method": m.get("method"),
                         "top1": m.get("top1"), "silhouette": m.get("silhouette"),
                         "recall@1": r["recall@1"], "recall@5": r["recall@5"],
                         "mAP": r["mAP"], **cl})
            print(f"  done {exp}")
    (ROOT / "results" / "retrieval_eval.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    hdr = ["exp", "method", "top1", "silh", "R@1", "R@5", "mAP", "purity", "nmi"]
    print("\n" + " | ".join(f"{h:>12s}" for h in hdr))
    last = None
    for row in rows:
        if row["group"] != last:
            print(f"# {row['group']}")
            last = row["group"]

        def fmt(v):
            return f"{v:.3f}" if isinstance(v, (int, float)) else "-"

        vals = [row["exp"][:12], (row["method"] or "-")[:12], fmt(row["top1"]),
                fmt(row["silhouette"]), fmt(row["recall@1"]), fmt(row["recall@5"]),
                fmt(row["mAP"]), fmt(row["purity"]), fmt(row["nmi"])]
        print(" | ".join(f"{v:>12s}" for v in vals))


if __name__ == "__main__":
    main()
