"""합성 매니폴드 3종: Interlocking Moons, Concentric Circles, Swiss Roll."""

import numpy as np
from sklearn.datasets import make_circles, make_moons, make_swiss_roll
from sklearn.model_selection import train_test_split


def make_moons_ds(n: int = 2000, noise: float = 0.08, seed: int = 42):
    x, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    return x.astype("float32"), y.astype("int64")


def make_circles_ds(n: int = 2000, noise: float = 0.06, factor: float = 0.5, seed: int = 42):
    x, y = make_circles(n_samples=n, noise=noise, factor=factor, random_state=seed)
    return x.astype("float32"), y.astype("int64")


def make_swiss_roll_ds(n: int = 2000, noise: float = 0.6, seed: int = 42, n_classes: int = 3):
    """3D 스위스롤의 나선 단면(x-z 평면)을 취하고, 나선 파라미터 t를 n_classes
    구간으로 잘라 레이블링 — 팔이 서로 감싸는 비선형 경계가 된다."""
    x3, t = make_swiss_roll(n_samples=n, noise=noise, random_state=seed)
    x = x3[:, [0, 2]].astype("float32")
    edges = np.quantile(t, np.linspace(0, 1, n_classes + 1))
    y = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, n_classes - 1)
    return x, y.astype("int64")


_MAKERS = {
    "moons": make_moons_ds,
    "circles": make_circles_ds,
    "swiss": make_swiss_roll_ds,
}


def get_synthetic(name: str, n: int = 2000, seed: int = 42, test_size: float = 0.3):
    """표준화(평균 0, 표준편차 1) + 층화 train/test 분할.

    Returns dict: x_train, y_train, x_test, y_test, mean, std (원좌표 복원용).
    """
    x, y = _MAKERS[name](n=n, seed=seed)
    mean, std = x.mean(0, keepdims=True), x.std(0, keepdims=True)
    x = (x - mean) / std
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )
    return {
        "x_train": x_tr,
        "y_train": y_tr,
        "x_test": x_te,
        "y_test": y_te,
        "mean": mean,
        "std": std,
    }


def sample_uniform_ood(n: int, low: float = -3.0, high: float = 3.0, seed: int = 0, dim: int = 2):
    """표준화 좌표계에서 균일 분포 OOD 샘플 (합성 실험의 에너지 지형 확인용)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(n, dim)).astype("float32")
