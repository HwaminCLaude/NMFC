"""plan7 ④⑤ — 검증쌍 AUC + few-shot 클래스 증분 곡선 (재학습 0, 캐시 embeddings.npz만).

④ 검증쌍 AUC: 테스트 임베딩에서 balanced 쌍(동/이 클래스)을 뽑아 코사인 점수로 same/diff 판정 AUC.
   "두 이미지가 같은 클래스인가?" — 얼굴인증·re-ID·상품매칭의 핵심 능력.
⑤ few-shot NCM: 프로토타입(지원 K장 평균)까지 최근접 분류. K-스윕 곡선 + 클래스 증분(S) 곡선.
   "참조 K장으로 클래스를 정의" — softmax가 구조적으로 약한 재학습 없는 클래스 추가의 데이터효율.

src/ 불변 — 순수 분석. 실행: PYTHONUTF8=1 .venv\\Scripts\\python.exe scripts/eval_fewshot_verify.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]

EXT = [  # plan8 — 외부 검색모델 (없으면 스킵)
    "ext_resnet50_frozen", "ext_vit_b16", "ext_convnext_base",
    "ext_clip_vitb32", "ext_dinov2_vits14",
]
DML = ["cifar100_normsoftmax", "cifar100_ms", "cifar100_proxyanchor"]  # plan9 강한 DML
DML_S = ["c100s_arcface", "c100s_normsoftmax", "c100s_ms",  # plan10 from-scratch DML
         "c100s_proxyanchor", "c100s_proto", "c100s_deepmfa"]
DML_LR3 = ["cifar100_arcface_lr3", "cifar100_normsoftmax_lr3", "cifar100_ms_lr3",  # plan11 lr 스윕
           "cifar100_proxyanchor_lr3", "cifar100_proxyanchor_plr", "cifar100_softmax_lr3"]
VAR = ["c100v_shtb", "c100v_shro", "c100v_shjb",  # v3·v4 변형 모듈 절제의 검색축 보강
       "c100v_shnb", "c100v_shjo", "c100v_shno"]
VERIF_EXPS = [
    "c100v_shrb", "cifar100_softmax", "c100s_shrb", "c100s_softmax",
    "cifar100_nmfc", "c100v_mhrb", "cifar100_arcface",
    "cifar100_proto", "cifar100_deepmfa", "cifar100_triplet",
] + EXT + DML + DML_S + VAR + DML_LR3
FEWSHOT = [  # (라벨, 실험명) — 집중 대상
    ("A shrb", "c100v_shrb"), ("A softmax", "cifar100_softmax"),
    ("A arcface", "cifar100_arcface"), ("A mhrb", "c100v_mhrb"),
    ("B shrb", "c100s_shrb"), ("B softmax", "c100s_softmax"),
] + [(e.replace("ext_", "ext "), e) for e in EXT] + [(e.replace("cifar100_", ""), e) for e in DML] \
  + [(e.replace("c100s_", "s "), e) for e in DML_S] + [(e.replace("c100v_", "v "), e) for e in VAR] \
  + [(e.replace("cifar100_", ""), e) for e in DML_LR3]
SHOTS = (1, 2, 5, 10, 20)
STEPS = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)


def unit(z):
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)


def by_class(y):
    classes = np.unique(y)
    idx = [np.where(y == c)[0] for c in classes]
    sizes = np.array([len(i) for i in idx])
    return classes, idx, sizes


def verification_auc(z, y, n_pairs=200_000, seed=42):
    """balanced same/diff 쌍 코사인 점수의 roc_auc."""
    rng = np.random.default_rng(seed)
    zc = unit(z)
    classes, idx, sizes = by_class(y)
    assert (sizes == sizes[0]).all(), "클래스 크기 불균형 — 벡터화 가정 위배"
    per = sizes[0]
    mat = np.stack(idx)  # (C, per) 클래스별 전역 인덱스

    # 양성: 클래스 하나 고르고 그 안에서 서로 다른 두 표본
    pc = rng.integers(0, len(classes), n_pairs)
    o1 = rng.integers(0, per, n_pairs)
    o2 = rng.integers(0, per - 1, n_pairs)
    o2 += (o2 >= o1)  # o1 != o2 보장
    pa, pb = mat[pc, o1], mat[pc, o2]
    pos = (zc[pa] * zc[pb]).sum(1)

    # 음성: 임의 두 표본 중 서로 다른 클래스만 (오버샘플 후 절단)
    neg = []
    while sum(len(x) for x in neg) < n_pairs:
        i = rng.integers(0, len(y), n_pairs)
        j = rng.integers(0, len(y), n_pairs)
        keep = y[i] != y[j]
        neg.append((zc[i[keep]] * zc[j[keep]]).sum(1))
    negs = np.concatenate(neg)[:n_pairs]

    scores = np.concatenate([pos, negs])
    labels = np.concatenate([np.ones(n_pairs), np.zeros(n_pairs)])
    return float(roc_auc_score(labels, scores))


def fewshot_kcurve(z, y, shots=SHOTS, episodes=5, seed=42):
    """K-스윕: 클래스당 지원 K장 프로토타입, 나머지 테스트 100-지 NCM 정확도."""
    rng = np.random.default_rng(seed)
    zc = unit(z)
    _, idx, _ = by_class(y)
    out = {}
    for k in shots:
        accs = []
        for _ in range(episodes):
            protos, q_z, q_lab = [], [], []
            for ci, ix in enumerate(idx):
                p = ix.copy()
                rng.shuffle(p)
                protos.append(zc[p[:k]].mean(0))
                q_z.append(zc[p[k:]])
                q_lab.append(np.full(len(p) - k, ci))
            proto = unit(np.stack(protos))
            qz = np.concatenate(q_z)
            ql = np.concatenate(q_lab)
            pred = (qz @ proto.T).argmax(1)
            accs.append(float((pred == ql).mean()))
        out[k] = float(np.mean(accs))
    return out


def incremental_curve(z, y, k=5, steps=STEPS, episodes=5, seed=42):
    """증분: K=5 고정, 갤러리 클래스 수 S를 키우며 S-지 정확도(고정 클래스 순서)."""
    rng = np.random.default_rng(seed)
    zc = unit(z)
    _, idx, _ = by_class(y)
    curves = {s: [] for s in steps}
    for _ in range(episodes):
        protos, q_z, q_lab = [], [], []
        for ci, ix in enumerate(idx):
            p = ix.copy()
            rng.shuffle(p)
            protos.append(zc[p[:k]].mean(0))
            q_z.append(zc[p[k:]])
            q_lab.append(np.full(len(p) - k, ci))
        proto = unit(np.stack(protos))
        qz = np.concatenate(q_z)
        ql = np.concatenate(q_lab)
        sims = qz @ proto.T  # (Nq, C)
        for s in steps:
            m = ql < s
            pred = sims[m, :s].argmax(1)
            curves[s].append(float((pred == ql[m]).mean()))
    return {s: float(np.mean(v)) for s, v in curves.items()}


def selftest():
    # 완전 분리: 동클래스 동일 벡터, 이클래스 직교 → AUC=1.0, few-shot=1.0
    z = np.repeat(np.eye(4), 25, axis=0).astype(np.float32)  # 100개, 4클래스 x 25
    y = np.repeat(np.arange(4), 25)
    assert abs(verification_auc(z, y, n_pairs=2000) - 1.0) < 1e-9
    assert abs(fewshot_kcurve(z, y, shots=(1,), episodes=2)[1] - 1.0) < 1e-9
    print("[selftest] verification/few-shot OK")


def load(exp):
    z = np.load(ROOT / "results" / exp / "embeddings.npz")
    m = json.loads((ROOT / "results" / exp / "metrics.json").read_text(encoding="utf-8"))
    return z["z_test"].astype(np.float32), z["y_test"].astype(int), m


def main():
    selftest()
    out = {"verification_auc": {}, "fewshot_kcurve": {}, "incremental_curve": {}}

    print("\n=== 4. 검증쌍 AUC (same/diff, balanced 20만+20만) ===")
    print(f"{'exp':22s} {'method':9s} {'verif_AUC':>10s}")
    for exp in VERIF_EXPS:
        if not (ROOT / "results" / exp / "embeddings.npz").exists():
            continue
        z, y, m = load(exp)
        auc = verification_auc(z, y)
        out["verification_auc"][exp] = {"method": m.get("method"), "auc": auc}
        print(f"{exp:22s} {str(m.get('method')):9s} {auc:>10.4f}")

    print("\n=== 5a. few-shot K-스윕 (100-지 NCM 정확도, 5에피소드 평균) ===")
    print(f"{'label':12s} " + " ".join(f"K={k:<5d}" for k in SHOTS))
    for label, exp in FEWSHOT:
        if not (ROOT / "results" / exp / "embeddings.npz").exists():
            continue
        z, y, _ = load(exp)
        kc = fewshot_kcurve(z, y)
        out["fewshot_kcurve"][exp] = {"label": label, **{str(k): v for k, v in kc.items()}}
        print(f"{label:12s} " + " ".join(f"{kc[k]:<7.3f}" for k in SHOTS))

    print("\n=== 5b. 클래스 증분 곡선 (K=5, S-지 정확도) ===")
    print(f"{'label':12s} " + " ".join(f"S={s:<4d}" for s in STEPS))
    for label, exp in FEWSHOT:
        if not (ROOT / "results" / exp / "embeddings.npz").exists():
            continue
        z, y, _ = load(exp)
        inc = incremental_curve(z, y)
        out["incremental_curve"][exp] = {"label": label, **{str(s): v for s, v in inc.items()}}
        print(f"{label:12s} " + " ".join(f"{inc[s]:<6.3f}" for s in STEPS))

    (ROOT / "results" / "fewshot_verify_eval.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[saved] results/fewshot_verify_eval.json")


if __name__ == "__main__":
    main()
