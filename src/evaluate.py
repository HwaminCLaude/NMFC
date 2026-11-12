"""평가 지표: Top-1/Top-5, Silhouette, OOD AUC."""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, silhouette_score


def topk_accuracy(logits: torch.Tensor, y: torch.Tensor, ks=(1, 5)) -> dict[str, float]:
    maxk = min(max(ks), logits.shape[1])
    pred = logits.topk(maxk, dim=1).indices
    correct = pred == y.view(-1, 1)
    out = {}
    for k in ks:
        if k <= logits.shape[1]:
            out[f"top{k}"] = correct[:, :k].any(dim=1).float().mean().item()
    return out


@torch.no_grad()
def embed_loader(encoder, loader, device, autocast: bool = True):
    """로더 전체를 임베딩 → (Z (N,m) fp32 CPU, y (N,))."""
    encoder.eval()
    zs, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast and device.type == "cuda"):
            z = encoder(x)
        zs.append(z.float().cpu())
        ys.append(y)
    return torch.cat(zs), torch.cat(ys)


def silhouette_of_embeddings(z, y, sample: int = 5000, seed: int = 42) -> float:
    """임베딩 군집 품질. 큰 데이터는 고정 시드 부분샘플."""
    z = np.asarray(z, dtype="float32")
    y = np.asarray(y)
    if len(z) > sample:
        idx = np.random.default_rng(seed).choice(len(z), size=sample, replace=False)
        z, y = z[idx], y[idx]
    return float(silhouette_score(z, y))


def ood_auc(scores_in: np.ndarray, scores_out: np.ndarray) -> float:
    """OOD 탐지 AUC — 점수가 클수록 OOD라는 규약 (E_total, 1-MSP 모두 해당)."""
    labels = np.concatenate([np.zeros(len(scores_in)), np.ones(len(scores_out))])
    scores = np.concatenate([scores_in, scores_out])
    return float(roc_auc_score(labels, scores))
