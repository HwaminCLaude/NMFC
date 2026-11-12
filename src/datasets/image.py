"""이미지 데이터셋 로더 — CIFAR-100/10, SVHN(test split만, OOD용).

Windows spawn 피클링을 위해 transform은 전부 torchvision 클래스 인스턴스로 구성
(lambda 금지). 사전학습 ResNet-50에 맞춰 ImageNet 정규화 통계를 쓴다.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .samplers import PKBatchSampler

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _train_transform(img_size: int):
    if img_size >= 96:  # 224 리사이즈 경로 — 사전학습 해상도에 맞춤
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(  # 32px 리허설 경로
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _test_transform(img_size: int):
    ops = []
    if img_size != 32:
        ops.append(transforms.Resize(img_size))
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


class IndexedDataset(torch.utils.data.Dataset):
    """base 데이터셋을 감싸 (x, y, sample_index)를 반환 — 온라인 뱅크 LOO 태그용.

    반드시 모듈 레벨 클래스(lambda·중첩 함수 금지) — Windows spawn 피클링 대비.
    """

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return x, y, i


class ClassSubsetDataset(torch.utils.data.Dataset):
    """base 데이터셋에서 keep_classes에 속한 샘플만 남긴다 — 원본 라벨 id 유지(리맵 없음).

    개방셋(open-set) 학습용: train split에서 novel 클래스 샘플만 제거하고, 라벨 공간은
    그대로 둔다(헤드 num_classes 불변, 희소 id는 NMFC forward_batch·softmax CE 모두 정상).
    반드시 모듈 레벨 클래스(lambda·중첩 함수 금지) — Windows spawn 피클링 대비.
    """

    def __init__(self, base, keep_classes):
        self.base = base
        keep = {int(c) for c in keep_classes}
        base_targets = np.asarray(base.targets if hasattr(base, "targets") else base.labels)
        self.indices = np.where(np.isin(base_targets, list(keep)))[0]
        # get_labels/PKBatchSampler 호환을 위해 .targets 노출 (원본 id)
        self.targets = base_targets[self.indices].tolist()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[int(self.indices[i])]


def select_seen_classes(total: int, n_seen: int, seed: int) -> np.ndarray:
    """total 클래스 중 n_seen개를 고정 시드로 무작위 선택 — 정렬된 원본 id 배열(결정적)."""
    assert 0 < n_seen < total, f"n_seen={n_seen}는 (0, {total}) 범위여야 함"
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=n_seen, replace=False))


def make_noisy_targets(targets, rate: float, seed: int, flip_classes, flip_pool) -> np.ndarray:
    """대칭 균등 라벨 노이즈 — flip_classes의 각 클래스에서 정확히 round(rate·n_c)개를
    자기 자신을 제외한 flip_pool로 균등 무작위 flip. 원본 배열 불변, 새 배열 반환.

    결정성 계약(plan14 §4.1): 단일 rng(np.random.default_rng(seed))를 flip_classes '오름차순'으로
    순회, 클래스당 (① flip 대상 인덱스 rng.choice(idx_c, n_flip, replace=False),
    ② 새 라벨 rng.choice(pool∖{c}, n_flip, replace=True)) 순서로만 소비.
    targets(디스크 고정 순서)·rate·seed·공간이 같으면 실현이 완전 동일 → 인스턴스 간 공유 보장.
    """
    orig = np.asarray(targets)
    noisy = orig.copy()
    rng = np.random.default_rng(seed)
    pool = np.asarray(flip_pool)
    for c in np.sort(np.asarray(flip_classes)):
        idx_c = np.where(orig == c)[0]
        n_flip = round(rate * len(idx_c))
        if n_flip == 0:
            continue
        chosen = rng.choice(idx_c, size=n_flip, replace=False)
        noisy[chosen] = rng.choice(pool[pool != c], size=n_flip, replace=True)
    return noisy


def resolve_label_noise(ln: dict | None, total: int | None, seen_classes=None) -> dict | None:
    """config {rate, seed, scope} → 실행 해석본. ln None → None(바이트 동일 게이트, rng 소비 0).

    scope="seen": flip 공간 = seen_classes(개방셋 분할과 결합 필수 — 침묵 실패 방지 assert).
    scope="all": flip 공간 = 전체 total 클래스(분할 전 오염 → seen↔novel 경계 침투 허용).
    """
    if ln is None:
        return None
    assert total is not None, "label_noise는 cifar100/cifar10 train에만 지원"
    scope = ln["scope"]
    assert scope in ("seen", "all"), f"label_noise.scope는 'seen'|'all'만 허용: {scope!r}"
    if scope == "seen":
        assert seen_classes is not None, "scope='seen'은 open_set(seen_classes)와 함께만 사용 가능"
        space = np.asarray(seen_classes)
    else:
        space = np.arange(total)
    return {
        "rate": float(ln["rate"]), "seed": int(ln["seed"]), "scope": scope,
        "flip_classes": space, "flip_pool": space,
        "seen_classes": None if seen_classes is None else np.asarray(seen_classes),
    }


def apply_label_noise(base_ds, noise: dict) -> dict:
    """base 데이터셋의 .targets를 노이즈판으로 교체(in-place, torchvision 원형 list[int] 유지).

    반드시 ClassSubsetDataset 래핑 '전'에 호출 — __getitem__과 .targets(PK 샘플러)가 한 번의
    교체로 자동 일치하고, scope='all'의 seen↔novel 멤버십 변화가 래핑 시점에 자동 반영된다.
    통계는 전부 python int(np.int64는 json.dump 실패). seen_classes 없으면 n_flipped만.
    """
    assert hasattr(base_ds, "targets"), "label_noise는 .targets 데이터셋만 지원(SVHN 미지원)"
    orig = np.asarray(base_ds.targets)
    noisy = make_noisy_targets(orig, noise["rate"], noise["seed"],
                               noise["flip_classes"], noise["flip_pool"])
    base_ds.targets = noisy.tolist()
    stats = {"n_flipped": int((orig != noisy).sum())}
    seen = noise.get("seen_classes")
    if seen is not None:
        orig_in = np.isin(orig, seen)
        noisy_in = np.isin(noisy, seen)
        flipped = orig != noisy
        stats.update({
            "n_dropped_seen_to_novel": int((orig_in & ~noisy_in).sum()),
            "n_contaminated_novel_to_seen": int((~orig_in & noisy_in).sum()),
            "n_mislabeled_in_train": int((orig_in & noisy_in & flipped).sum()),
            "train_size": int(noisy_in.sum()),
        })
    return stats


def get_image_datasets(name: str, root: str | Path, img_size: int, seen_classes=None,
                       label_noise=None):
    """(train_ds, test_ds, num_classes). SVHN은 test split만 (OOD 평가 전용).

    seen_classes(개방셋 학습)가 주어지면 train split을 해당 클래스로만 서브셋한다 — 원본
    라벨 공간·반환 num_classes 불변, test split은 전체 유지. None이면 기존과 완전 동일.
    label_noise(resolve_label_noise 해석본)가 주어지면 train base의 .targets를 래핑 '전'에
    교체(test는 절대 불변)하고 train_ds.noise_stats에 통계를 부착. None이면 바이트 동일.
    """
    root = str(root)
    if name == "cifar100":
        tr = datasets.CIFAR100(root, train=True, download=True, transform=_train_transform(img_size))
        te = datasets.CIFAR100(root, train=False, download=True, transform=_test_transform(img_size))
        stats = apply_label_noise(tr, label_noise) if label_noise is not None else None
        if seen_classes is not None:
            tr = ClassSubsetDataset(tr, seen_classes)
        if stats is not None:
            tr.noise_stats = stats
        return tr, te, 100
    if name == "cifar10":
        tr = datasets.CIFAR10(root, train=True, download=True, transform=_train_transform(img_size))
        te = datasets.CIFAR10(root, train=False, download=True, transform=_test_transform(img_size))
        stats = apply_label_noise(tr, label_noise) if label_noise is not None else None
        if seen_classes is not None:
            tr = ClassSubsetDataset(tr, seen_classes)
        if stats is not None:
            tr.noise_stats = stats
        return tr, te, 10
    if name == "svhn":
        assert label_noise is None, "SVHN은 label_noise 미지원(test split 전용)"
        te = datasets.SVHN(root, split="test", download=True, transform=_test_transform(img_size))
        return None, te, 10
    raise ValueError(f"알 수 없는 데이터셋: {name}")


def get_plain_train_dataset(name: str, root: str | Path, img_size: int, seen_classes=None,
                            label_noise=None):
    """증강 없는(test transform) train split — Fisher ratio 프로브·ReferenceBank용.

    seen_classes가 주어지면 해당 클래스로만 서브셋(개방셋 학습 분포용). None이면 기존과 동일.
    label_noise는 get_image_datasets와 동일 계약(같은 해석본이면 실현도 동일 — 순수 함수+시드).
    """
    root = str(root)
    if name == "cifar100":
        ds = datasets.CIFAR100(root, train=True, download=True, transform=_test_transform(img_size))
    elif name == "cifar10":
        ds = datasets.CIFAR10(root, train=True, download=True, transform=_test_transform(img_size))
    else:
        raise ValueError(f"알 수 없는 데이터셋: {name}")
    if label_noise is not None:
        apply_label_noise(ds, label_noise)
    if seen_classes is not None:
        ds = ClassSubsetDataset(ds, seen_classes)
    return ds


def get_labels(ds) -> np.ndarray:
    if hasattr(ds, "targets"):
        return np.asarray(ds.targets)
    return np.asarray(ds.labels)  # SVHN


def make_train_loader(
    train_ds,
    mode: str = "shuffle",
    batch_size: int = 256,
    p: int = 32,
    k: int = 8,
    num_workers: int = 4,
    seed: int = 42,
):
    """mode='shuffle'(CE 계열) 또는 'pk'(NMFC/Triplet/Deep-MFA)."""
    common = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    if mode == "pk":
        sampler = PKBatchSampler(get_labels(train_ds), p=p, k=k, seed=seed)
        return DataLoader(train_ds, batch_sampler=sampler, **common)
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, generator=g, **common
    )


def make_eval_loader(ds, batch_size: int = 256, num_workers: int = 4):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
