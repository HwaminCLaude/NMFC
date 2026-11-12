"""실험③ (plan6) — softmax 인코더 + 참조뱅크로 novel 분류.

캐시된 c100v_softmax_open 체크포인트의 인코더를 그대로 쓰되, 추론만 참조뱅크(코어셋)로 바꿔
학습 안 한 novel 클래스를 분류한다. 재학습 없음 — "뱅크 확장성이 에너지 학습 덕이냐 인코더 덕이냐" 분리.

실행: PYTHONUTF8=1 .venv\\Scripts\\python.exe scripts/eval_softmax_bank.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.image import (  # noqa: E402
    get_image_datasets,
    get_plain_train_dataset,
    make_eval_loader,
    select_seen_classes,
)
from src.evaluate import embed_loader  # noqa: E402
from src.models.encoders import ResNet50Encoder  # noqa: E402
from src.models.heads_v3 import VariantNMFCHead  # noqa: E402
from src.utils.misc import get_device  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# 인코더를 빌려올 캐시 실험 (argv로 교체 가능 — 공정 비교: 같은 뱅크·규칙으로 shrb 인코더도 평가)
SRC_EXP = sys.argv[1] if len(sys.argv) > 1 else "c100v_softmax_open"
OUT_EXP = SRC_EXP + "_bank"
N_SEEN, SPLIT_SEED, IMG_SIZE, EMBED_DIM = 80, 0, 224, 128


def main():
    device = get_device()
    ckpt = torch.load(ROOT / "checkpoints" / SRC_EXP / "ckpt.pt",
                      map_location=device, weights_only=False)
    encoder = ResNet50Encoder(EMBED_DIM, pretrained=False).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()

    root = ROOT / "data" / "raw"
    _, test_ds, _ = get_image_datasets("cifar100", root, IMG_SIZE)
    plain_train = get_plain_train_dataset("cifar100", root, IMG_SIZE)  # 전체 100클래스
    z_test, y_test = embed_loader(encoder, make_eval_loader(test_ds, num_workers=4), device)
    z_train, y_train = embed_loader(encoder, make_eval_loader(plain_train, num_workers=4), device)

    # c100v_shrb와 동일 설정으로 코어셋 뱅크 fit (에너지 계산기만 재사용)
    head = VariantNMFCHead(
        energy="softmin", logit="hinge_temp", lam=0.5, k1=3, k2=32,
        tau_min=0.01, m_quantile=0.75, n_coreset=64, n_ref=64, detach_pi=False,
    ).to(device)
    head.fit_coreset_bank(z_train.to(device), y_train.to(device), seed=42)

    # predict = argmin_k E⁺_k (힌지 없이 최근접 양성 에너지)
    preds = []
    with torch.no_grad():
        for s in range(0, len(z_test), 1024):
            e_pos, _ = head._bank_energies(z_test[s:s + 1024].to(device))
            preds.append(head.bank_classes[e_pos.argmin(1)].cpu())
    pred = torch.cat(preds).numpy()

    seen = select_seen_classes(100, N_SEEN, SPLIT_SEED)
    y = y_test.numpy()
    seen_mask = np.isin(y, seen.tolist())
    novel_mask = ~seen_mask
    correct = pred == y
    metrics = {
        "runner": "openset_bank_eval",
        "source_encoder": SRC_EXP,
        "n_seen": N_SEEN, "split_seed": SPLIT_SEED,
        "open_top1_all": float(correct.mean()),
        "open_top1_seen": float(correct[seen_mask].mean()),
        "open_top1_novel": float(correct[novel_mask].mean()),
        "predict": "argmin_E_pos (bank, no hinge)",
    }
    out = ROOT / "results" / OUT_EXP
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
