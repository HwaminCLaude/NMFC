"""라벨 노이즈(plan14) 테스트 — make_noisy_targets/resolve_label_noise/apply_label_noise.

torchvision 다운로드 없이 페이크 데이터셋으로 검증(CPU, 수 초).
페이크: 10클래스 × 100샘플, rate 0.05 → 클래스당 정확 5개 flip, seen = 임의 8클래스.
"""

import json

import numpy as np
import pytest

from src.datasets.image import (
    ClassSubsetDataset,
    apply_label_noise,
    get_labels,
    make_noisy_targets,
    resolve_label_noise,
)
from src.datasets.samplers import PKBatchSampler

RATE, NSEED = 0.05, 0
N_CLASSES, PER_CLASS = 10, 100
SEEN = [0, 1, 2, 4, 5, 7, 8, 9]  # 임의 8클래스 (novel = 3, 6)


class _FakeDS:
    """targets/__getitem__만 가진 최소 데이터셋 (torchvision 다운로드 회피).

    torchvision CIFAR와 동일하게 .targets는 가변 list[int] — apply_label_noise의
    재할당(base.targets = noisy.tolist())·읽기(np.asarray) 계약과 형태 일치.
    """

    def __init__(self, targets):
        self.targets = list(targets)
        self.data = np.arange(len(targets))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.data[i], self.targets[i]


def _targets():
    return np.array([i % N_CLASSES for i in range(N_CLASSES * PER_CLASS)])


def test_deterministic():
    orig = _targets()
    before = orig.copy()
    space = np.arange(N_CLASSES)
    a = make_noisy_targets(orig, RATE, NSEED, space, space)
    b = make_noisy_targets(orig, RATE, NSEED, space, space)
    assert np.array_equal(a, b)                          # 같은 인자 = 같은 실현
    assert not np.array_equal(a, make_noisy_targets(orig, RATE, 1, space, space))  # 시드 민감
    assert np.array_equal(orig, before)                  # 원본 불변(순수성)


def test_exact_flip_count_per_class():
    orig = _targets()
    space = np.arange(N_CLASSES)
    noisy = make_noisy_targets(orig, RATE, NSEED, space, space)
    for c in space:
        n_c = int((orig == c).sum())
        assert ((orig == c) & (orig != noisy)).sum() == round(RATE * n_c)  # 클래스당 정확 5개


def test_no_self_flip_pool_respected():
    orig = _targets()
    flip_classes = np.asarray(SEEN)
    noisy = make_noisy_targets(orig, RATE, NSEED, flip_classes, flip_classes)
    moved = orig != noisy
    assert np.isin(noisy[moved], flip_classes).all()      # 새 라벨 ∈ flip_pool
    assert (noisy[moved] != orig[moved]).all()            # 자기 자신 flip 금지
    outside = ~np.isin(orig, flip_classes)
    assert np.array_equal(orig[outside], noisy[outside])  # flip_classes 밖 불변


def test_scope_seen_invariants():
    orig = _targets()
    noise = resolve_label_noise({"rate": RATE, "seed": NSEED, "scope": "seen"},
                                N_CLASSES, seen_classes=SEEN)
    noisy = make_noisy_targets(orig, noise["rate"], noise["seed"],
                               noise["flip_classes"], noise["flip_pool"])
    assert np.isin(noisy, SEEN).sum() == np.isin(orig, SEEN).sum()  # train 크기 불변
    novel_rows = ~np.isin(orig, SEEN)
    assert np.array_equal(orig[novel_rows], noisy[novel_rows])      # novel 행 완전 불변


def test_scope_all_stats():
    orig = _targets()
    ds = _FakeDS(orig)
    noise = resolve_label_noise({"rate": RATE, "seed": NSEED, "scope": "all"},
                                N_CLASSES, seen_classes=SEEN)
    stats = apply_label_noise(ds, noise)
    noisy = np.asarray(ds.targets)
    orig_in, noisy_in = np.isin(orig, SEEN), np.isin(noisy, SEEN)
    # 상수 하드코딩 금지 — orig/noisy 재계산으로 대조
    assert stats["n_flipped"] == int((orig != noisy).sum()) \
        == sum(round(RATE * (orig == c).sum()) for c in range(N_CLASSES))
    assert stats["n_dropped_seen_to_novel"] == int((orig_in & ~noisy_in).sum())
    assert stats["n_contaminated_novel_to_seen"] == int((~orig_in & noisy_in).sum())
    assert stats["n_mislabeled_in_train"] == int((orig_in & noisy_in & (orig != noisy)).sum())
    assert stats["train_size"] == int(orig_in.sum()) \
        - stats["n_dropped_seen_to_novel"] + stats["n_contaminated_novel_to_seen"]


def test_shared_realization():
    orig = _targets()
    ds1, ds2 = _FakeDS(orig), _FakeDS(orig)
    noise = resolve_label_noise({"rate": RATE, "seed": NSEED, "scope": "all"},
                                N_CLASSES, seen_classes=SEEN)
    apply_label_noise(ds1, noise)
    apply_label_noise(ds2, noise)
    assert ds1.targets == ds2.targets  # train_ds↔plain_train_ds↔평가A 3중 정합의 핵심 계약


def test_subset_follows_noisy():
    orig = _targets()
    ds = _FakeDS(orig)
    noise = resolve_label_noise({"rate": RATE, "seed": NSEED, "scope": "all"},
                                N_CLASSES, seen_classes=SEEN)
    stats = apply_label_noise(ds, noise)
    sub = ClassSubsetDataset(ds, SEEN)
    assert len(sub) == stats["train_size"]  # 멤버십이 노이즈 라벨 기준
    sub_targets = np.asarray(sub.targets)
    assert np.isin(sub_targets, SEEN).all()
    for i in range(0, len(sub), 97):
        _, y = sub[i]
        assert int(y) == int(sub_targets[i])  # __getitem__ 라벨 == .targets 스냅샷


def test_resolver_gate():
    assert resolve_label_noise(None, N_CLASSES, seen_classes=SEEN) is None
    with pytest.raises(AssertionError):
        resolve_label_noise({"rate": RATE, "seed": NSEED, "scope": "seen"}, N_CLASSES)
    with pytest.raises(AssertionError):
        resolve_label_noise({"rate": RATE, "seed": NSEED, "scope": "bogus"}, N_CLASSES)


def test_stats_json_safe():
    ds = _FakeDS(_targets())
    noise = resolve_label_noise({"rate": RATE, "seed": NSEED, "scope": "all"},
                                N_CLASSES, seen_classes=SEEN)
    stats = apply_label_noise(ds, noise)
    assert all(type(v) is int for v in stats.values())  # np.int64는 json.dump 실패
    json.dumps(stats)


def test_pk_sampler_on_noisy_subset():
    ds = _FakeDS(_targets())
    noise = resolve_label_noise({"rate": RATE, "seed": NSEED, "scope": "all"},
                                N_CLASSES, seen_classes=SEEN)
    apply_label_noise(ds, noise)
    sub = ClassSubsetDataset(ds, SEEN)
    sampler = PKBatchSampler(get_labels(sub), p=4, k=4, seed=0)
    assert len(sampler.classes) == len(SEEN)  # usable 클래스 == seen (rate 5%로는 고갈 불가)
    batch = next(iter(sampler))
    assert len(batch) == 4 * 4
    labels = np.asarray(sub.targets)[np.asarray(batch)]
    assert all((labels == c).sum() == 4 for c in np.unique(labels))  # P×K 균형
