"""plan12 확장 — near-OOD 사후 평가 (재학습 0, ckpt.pt + embeddings.npz만 사용).

기존 OOD 평가(metrics.json의 ood_auc)는 far-OOD(CIFAR-10 -> SVHN)만 측정한다. 이 스크립트는 같은 모델·같은 동결
추론 상태(코어셋 뱅크·sigma^2·tau·m)로 near-OOD(CIFAR-10 -> CIFAR-100 test, OpenOOD 관행 쌍)를
사후 채점한다. in-분포 점수는 학습 시 저장된 embeddings.npz의 ood_scores_in을 재사용하므로
far/near AUROC가 완전히 같은 in-점수 축 위에서 비교된다.

정합성 가드(침묵 실패 방지):
  1) 저장된 z_test를 복원 헤드로 재채점 -> ood_scores_in과 최대오차 비교 (채점 경로 검증)
  2) CIFAR-10 test 앞 256장을 재임베딩 -> 저장 z_test와 비교 (인코더 복원 검증, 장치별 오차 보고)
  3) 저장 점수로 far AUC 재계산 -> metrics.json ood_auc와 일치 확인

실행: PYTHONUTF8=1 .venv\\Scripts\\python.exe scripts/eval_near_ood.py [--device cpu|cuda] [--exps A B]
출력: results/near_ood.json (+ 실험별 results/<exp>/near_ood.npz). 미완료 실험 자동 스킵.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import ood_auc                      # noqa: E402
from src.datasets.image import get_image_datasets, make_eval_loader  # noqa: E402
from src.models.baselines import LinearHead           # noqa: E402
from src.models.encoders import ResNet50Encoder       # noqa: E402
from src.models.heads_v3 import VariantNMFCHead       # noqa: E402
from src.utils.experiments import exp_paths           # noqa: E402

SEEDS = [42, 0, 1, 2, 3]
BASES = ["ood10v_shrb", "ood10v_mhrb", "ood_cifar10_softmax_lr3"]
NEAR_DATASET = "cifar100"


def exp_name(base, seed):
    return base if seed == 42 else f"{base}_s{seed}"


def build_head(spec, npz, device):
    base_cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))
    nm = {**base_cfg["nmfc"], **spec.get("nmfc", {})}
    head = VariantNMFCHead(
        energy=str(npz["head_energy"]), logit=str(npz["head_logit"]), lam=nm["lam"],
        k1=nm["k1"], k2=nm["k2"], tau_min=nm["tau_min"], m_quantile=nm["m_quantile"],
        n_coreset=nm["n_coreset"], n_ref=nm["n_ref"], detach_pi=nm["detach_pi"],
    )
    head.load_bank_state(npz, device=device)
    return head


@torch.no_grad()
def embed(encoder, loader, device):
    outs = []
    for x, _ in loader:
        outs.append(encoder(x.to(device)).float().cpu())
    return torch.cat(outs)


def eval_exp(exp, device, num_workers):
    paths = exp_paths(exp)
    ckpt_f = paths["ckpt"] / "ckpt.pt"
    emb_f = paths["result"] / "embeddings.npz"
    met_f = paths["result"] / "metrics.json"
    if not (ckpt_f.exists() and emb_f.exists() and met_f.exists()):
        return None
    metrics = json.loads(met_f.read_text(encoding="utf-8"))
    with np.load(emb_f) as z:
        npz = dict(z)
    if "ood_scores_in" not in npz:
        return None
    ckpt = torch.load(ckpt_f, map_location="cpu", weights_only=True)
    spec = ckpt["spec"]

    encoder = ResNet50Encoder(spec["embed_dim"], pretrained=False)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.to(device).eval()

    scores_in = npz["ood_scores_in"]
    z_test = torch.as_tensor(npz["z_test"])
    if spec["method"] == "nmfc":
        head = build_head(spec, npz, device)
        def score(z):
            return head.ood_score(z.to(device)).cpu().numpy()
    else:
        n_classes = ckpt["head"]["fc.weight"].shape[0]
        head = LinearHead(spec["embed_dim"], n_classes)
        head.load_state_dict(ckpt["head"])
        head.to(device).eval()
        @torch.no_grad()
        def score(z):
            probs = torch.cat([
                torch.softmax(head(z[s: s + 4096].to(device)), dim=1).max(1).values.cpu()
                for s in range(0, len(z), 4096)])
            return (1.0 - probs).numpy()

    # 가드 1: 채점 경로 재현 (저장 z_test -> 저장 in-점수). 점수 절대오차는 장치(CPU/GPU) 부동소수
    # 차이로 스케일(~70) 대비 ~2e-4 상대 수준까지 정상 — 판정은 AUC 불변성(가드 3)으로 한다.
    rescored = score(z_test)
    score_err = float(np.max(np.abs(rescored - scores_in)))
    # 가드 3: far AUC 재현 — 저장 in-점수 기준(↔metrics) + 재채점 in-점수 기준(채점 경로 등가성)
    far_auc = ood_auc(scores_in, npz["ood_scores_out"])
    far_ref = metrics["ood_auc"]
    far_auc_rescored = ood_auc(rescored, npz["ood_scores_out"])

    # 가드 2: 인코더 복원 (in-분포 test 앞 256장 재임베딩)
    root = ROOT / "data" / "raw"
    _, in_test, _ = get_image_datasets(spec["dataset"], root, spec["img_size"])
    sub_loader = make_eval_loader(torch.utils.data.Subset(in_test, list(range(256))),
                                  num_workers=num_workers)
    z_sub = embed(encoder, sub_loader, device)
    enc_err = float(np.max(np.abs(z_sub.numpy() - npz["z_test"][:256])))

    # near-OOD 채점 (CIFAR-100 test 10k)
    _, near_ds, _ = get_image_datasets(NEAR_DATASET, root, spec["img_size"])
    near_loader = make_eval_loader(near_ds, num_workers=num_workers)
    z_near = embed(encoder, near_loader, device)
    scores_near = score(z_near)
    near_auc = ood_auc(scores_in, scores_near)
    near_auc_rescored = ood_auc(rescored, scores_near)  # in-점수도 현 파이프라인으로 통일한 교차판

    np.savez(paths["result"] / "near_ood.npz", scores_near=scores_near)
    row = {
        "method": spec["method"], "ood_score": metrics.get("ood_score"),
        "far_dataset": spec.get("ood_dataset"), "far_auc": far_auc, "far_auc_metrics": far_ref,
        "near_dataset": NEAR_DATASET, "near_auc": near_auc, "near_auc_rescored_in": near_auc_rescored,
        "guard_score_maxerr": score_err, "guard_encoder_maxerr": enc_err,
        "guard_far_auc_rescored": far_auc_rescored,
        "device": str(device),
    }
    flag = " [!]" if (abs(far_auc - far_ref) > 1e-6 or abs(far_auc_rescored - far_ref) > 1e-3
                     or abs(near_auc - near_auc_rescored) > 1e-3) else ""
    print(f"{exp}: far={far_auc:.4f} near={near_auc:.4f} "
          f"(score_err={score_err:.2e}, enc_err={enc_err:.2e}){flag}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--exps", nargs="*", default=None, help="실험명 직접 지정(기본: BASES x SEEDS)")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=8, help="CPU 스레드 상한(학습 큐 보호)")
    args = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cpu":
        torch.set_num_threads(args.threads)

    exps = args.exps or [exp_name(b, s) for b in BASES for s in SEEDS]
    out_f = ROOT / "results" / "near_ood.json"
    out = json.loads(out_f.read_text(encoding="utf-8")) if out_f.exists() else {}
    for exp in exps:
        row = eval_exp(exp, device, args.num_workers)
        if row is None:
            print(f"{exp}: 미완료/산출물 없음 -> 스킵")
            continue
        out[exp] = row
        out_f.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"saved -> {out_f}")


if __name__ == "__main__":
    main()
