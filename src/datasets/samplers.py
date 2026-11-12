"""PK 균형 배치 샘플러 — 배치마다 P개 클래스 × K개 샘플.

NMFC(배치 내 에너지에 클래스당 양성 이웃 K-1개 보장)와 Triplet(양성 쌍 보장)이
공유한다. P>=2, K>=2가 아니면 NMFC 에너지·삼중항이 정의되지 않으므로 assert.
"""

import math

import numpy as np
from torch.utils.data import Sampler


class PKBatchSampler(Sampler[list[int]]):
    def __init__(self, labels, p: int, k: int, iters: int | None = None, seed: int = 42):
        labels = np.asarray(labels)
        classes, counts = np.unique(labels, return_counts=True)
        usable = classes[counts >= k]
        if len(usable) < 2 or k < 2:
            raise ValueError(f"PK 샘플러는 P>=2, K>=2 필요 (사용 가능 클래스 {len(usable)}, K={k})")
        self.p = min(p, len(usable))
        self.k = k
        self.classes = usable
        self.class_indices = {c: np.where(labels == c)[0] for c in usable}
        self.iters = iters if iters is not None else math.ceil(len(labels) / (self.p * k))
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        for _ in range(self.iters):
            cs = self.rng.choice(self.classes, size=self.p, replace=False)
            batch = np.concatenate(
                [self.rng.choice(self.class_indices[c], size=self.k, replace=False) for c in cs]
            )
            yield batch.tolist()

    def __len__(self):
        return self.iters
