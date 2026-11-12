"""보정(calibration)·과신(overconfidence) 평가 — 재학습 0, 캐시 ckpt/embeddings만 사용.

동기: "소프트맥스의 전역 선형 로짓은 국소 기하를 담지 못해 과신한다. NMFC의 국소 에너지
로짓은 그렇지 않다"는 가설을 직접 시험한다. 이전에는 이 축을 한 번도 측정하지 않았다.

측정 지표 (온도 교란을 분리하도록 3층으로 구성):
  1. 과신 격차 = mean(max prob) - accuracy  ... 가장 직관적인 '과신' 정의
  2. ECE(15빈) raw / 온도스케일링(TS) 후    ... TS 후 잔존 ECE가 '구조적' 보정 차이
  3. 오분류 탐지 AUROC · AURC · E-AURC      ... 스케일 불변(확신도 '순서'만 사용)

**핵심 주의**: NMFC는 학습 온도 tau를 갖고 소프트맥스 CE는 갖지 않으므로, raw ECE 비교는
구조가 아니라 온도를 재는 것에 가깝다. 따라서 결론은 (a) TS 후 ECE와 (b) 스케일 불변인
오분류 탐지 AUROC / E-AURC 에 근거해야 한다. raw 수치는 참고로만 병기한다.

TS는 테스트 분할을 절반으로 나눠(고정 시드) 앞쪽에서 T를 적합하고 뒤쪽에서 보고한다
(동일 분할을 모든 방법에 적용). accuracy 는 metrics.json 과 대조해 자기검증한다.

실행: PYTHONUTF8=1 .venv\\Scripts\\python.exe scripts/eval_calibration.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # scripts/ 직접 실행 시 src 패키지 해석
SEED = 42
N_BINS = 15

# (실험명, 표시라벨) — 사전학습 40ep 트랙만(체제 혼합 금지).
EXPS = [
    ("c100v_shrb", "NMFC shrb (제안)"),
    ("c100v_mhrb", "NMFC mhrb"),
    ("cifar100_nmfc", "NMFC v1"),
    ("cifar100_softmax", "Softmax CE (lr1e-3)"),
    ("cifar100_softmax_lr3", "Softmax CE (lr3e-4)"),
    ("cifar100_arcface", "ArcFace (lr1e-3)"),
    ("cifar100_arcface_lr3", "ArcFace (lr3e-4)"),
    ("cifar100_normsoftmax_lr3", "NormSoftmax (lr3e-4)"),
    ("cifar100_proxyanchor_lr3", "Proxy-Anchor (lr3e-4)"),
    ("cifar100_proxyanchor_plr", "Proxy-Anchor +plr"),
    ("cifar100_ms_lr3", "Multi-Similarity (lr3e-4)"),
    ("cifar100_proto", "Prototypical"),
    ("cifar100_deepmfa", "Deep-MFA"),
]


# ---------------------------------------------------------------- 로짓 복원
def nmfc_logits(exp, z, device):
    """VariantNMFCHead(shrb 등) 또는 v1 NMFCHead의 추론 로짓을 캐시에서 복원."""
    from src.models import VariantNMFCHead

    npz = np.load(ROOT / "results" / exp / "embeddings.npz")
    if "head_energy" not in npz.files:
        # v1 NMFCHead(ReferenceBank: bank_mu_pos/neg 규약)는 로짓 복원 경로가 달라 제외한다.
        # 본 분석의 관심은 v3 계열 국소 에너지 로짓 대 선형/코사인 헤드의 대조이므로 손실이 없다.
        return None
    head = VariantNMFCHead(
        energy=str(npz["head_energy"]), logit=str(npz["head_logit"]), lam=0.5,
        k1=3, k2=32, tau_min=0.01, m_quantile=0.75, n_coreset=64, n_ref=64, detach_pi=False,
    )
    head.load_bank_state({k: npz[k] for k in npz.files if k not in ("z_test", "y_test")}, device=device)
    return head.forward_bank(z)


def baseline_logits(exp, z, device):
    """선형/코사인 헤드의 추론 로짓 (학습 시 마진은 추론에 미적용 — 각 헤드 forward 규약 그대로)."""
    ck = torch.load(ROOT / "checkpoints" / exp / "ckpt.pt", map_location=device, weights_only=False)
    h, spec = ck["head"], ck["spec"]
    method = spec.get("method", "")
    if "fc.weight" in h:  # softmax: 선형 헤드
        return z @ h["fc.weight"].to(device).T + h["fc.bias"].to(device)
    if "weight" in h:  # arcface / normsoftmax: 코사인 × 스케일
        w = F.normalize(h["weight"].to(device))
        scale = spec.get("s", 30.0) if method == "arcface" else spec.get("scale", 20.0)
        return scale * (F.normalize(z) @ w.T)
    if "proxies" in h:  # proxyanchor: 코사인(스케일 없음)
        return F.normalize(z) @ F.normalize(h["proxies"].to(device)).T
    raise ValueError(f"{exp}: 알 수 없는 헤드 구조 {list(h.keys())}")


def get_logits(exp, device):
    npz = np.load(ROOT / "results" / exp / "embeddings.npz")
    z = torch.as_tensor(npz["z_test"], dtype=torch.float32, device=device)
    y = torch.as_tensor(npz["y_test"].astype(int), dtype=torch.long, device=device)
    m = json.loads((ROOT / "results" / exp / "metrics.json").read_text(encoding="utf-8"))
    if m.get("method") == "nmfc":
        return nmfc_logits(exp, z, device), y, m
    if m.get("method") in ("proto", "ms", "triplet"):
        return None, y, m  # 무헤드(프로토타입 NCM) — 확률 개념이 달라 제외
    return baseline_logits(exp, z, device), y, m


# ---------------------------------------------------------------- 지표
def ece(conf, correct, n_bins=N_BINS):
    """Expected Calibration Error (등간격 빈)."""
    edges = np.linspace(0, 1, n_bins + 1)
    e, n = 0.0, len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        e += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def reliability(conf, correct, n_bins=N_BINS):
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        out.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                    "conf": float(conf[m].mean()) if m.sum() else None,
                    "acc": float(correct[m].mean()) if m.sum() else None})
    return out


def aurc(conf, correct):
    """AURC(낮을수록 좋음)와 E-AURC(정확도 차이를 보정한 초과분)."""
    order = np.argsort(-conf)
    err = 1.0 - correct[order]
    risks = np.cumsum(err) / np.arange(1, len(err) + 1)
    a = float(risks.mean())
    r = 1.0 - correct.mean()  # 전체 오류율
    # 완벽한 확신도 순서일 때의 AURC(오답이 전부 뒤로 밀린 경우)
    opt_err = np.concatenate([np.zeros(int(round((1 - r) * len(err)))), np.ones(len(err) - int(round((1 - r) * len(err))))])
    opt = float((np.cumsum(opt_err) / np.arange(1, len(err) + 1)).mean())
    return a, float(a - opt)


def fit_temperature(logits, y):
    """NLL 최소화로 스칼라 온도 T 적합(LBFGS)."""
    log_t = torch.zeros(1, device=logits.device, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def selftest():
    """완벽 보정(정확도=확신도) 구성에서 ECE≈0, 무작위 확신도에서 오분류 AUROC≈0.5."""
    rng = np.random.default_rng(0)
    correct = rng.random(20000) < 0.8
    conf = np.where(correct, 0.8, 0.8)  # 모든 빈에서 conf=acc=0.8
    assert ece(conf, correct.astype(float)) < 0.02, ece(conf, correct.astype(float))
    auc = roc_auc_score(1 - correct, -rng.random(20000))
    assert abs(auc - 0.5) < 0.02
    print("[selftest] ECE/AUROC OK")


def main():
    selftest()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    rng = np.random.default_rng(SEED)
    rows, rel_out = [], {}
    perm = None

    for exp, label in EXPS:
        if not (ROOT / "results" / exp / "embeddings.npz").exists():
            print(f"  skip {exp} (없음)")
            continue
        logits, y, m = get_logits(exp, device)
        if logits is None:
            print(f"  skip {exp} (무헤드 방법 — 확률 정의 불가)")
            continue
        n = logits.shape[0]
        if perm is None:  # 모든 방법에 동일한 cal/eval 분할 적용
            perm = rng.permutation(n)
        cal, ev = torch.as_tensor(perm[: n // 2], device=device), torch.as_tensor(perm[n // 2:], device=device)

        T = fit_temperature(logits[cal].detach(), y[cal])
        for tag, lg in (("raw", logits), ("ts", logits / T)):
            p = F.softmax(lg[ev], dim=1)
            conf, pred = p.max(1)
            corr = (pred == y[ev]).float()
            c, k = conf.cpu().numpy(), corr.cpu().numpy()
            acc = float(k.mean())
            a, ea = aurc(c, k)
            row = {"exp": exp, "label": label, "variant": tag, "T": T, "acc": acc,
                   "mean_conf": float(c.mean()), "overconf_gap": float(c.mean() - acc),
                   "ece": ece(c, k), "aurc": a, "e_aurc": ea,
                   "mis_auroc": float(roc_auc_score(1 - k, -c)) if 0 < k.mean() < 1 else None}
            rows.append(row)
            if tag == "raw":
                rel_out[exp] = {"label": label, "bins": reliability(c, k)}
                ref = m.get("top1")
                if ref is not None and abs(acc - ref) > 0.03:
                    print(f"  [주의] {exp}: 복원 정확도 {acc:.4f} vs metrics.json {ref:.4f} — 헤드 복원 확인 필요")
        print(f"  done {exp} (T={T:.3f})")

    (ROOT / "results" / "calibration_eval.json").write_text(
        json.dumps({"rows": rows, "reliability": rel_out}, ensure_ascii=False, indent=2), encoding="utf-8")

    hdr = ["label", "acc", "conf", "과신격차", "ECE", "ECE(TS)", "T", "오분류AUROC", "E-AURC"]
    print("\n" + " | ".join(f"{h:>16s}" if i == 0 else f"{h:>9s}" for i, h in enumerate(hdr)))
    raw = {r["exp"]: r for r in rows if r["variant"] == "raw"}
    ts = {r["exp"]: r for r in rows if r["variant"] == "ts"}
    for exp, _ in EXPS:
        if exp not in raw:
            continue
        r, t = raw[exp], ts[exp]
        vals = [r["label"][:16], f"{r['acc']:.4f}", f"{r['mean_conf']:.4f}", f"{r['overconf_gap']:+.4f}",
                f"{r['ece']:.4f}", f"{t['ece']:.4f}", f"{r['T']:.3f}",
                f"{r['mis_auroc']:.4f}", f"{r['e_aurc']:.5f}"]
        print(" | ".join(f"{v:>16s}" if i == 0 else f"{v:>9s}" for i, v in enumerate(vals)))
    print("\n[saved] results/calibration_eval.json")


if __name__ == "__main__":
    main()
