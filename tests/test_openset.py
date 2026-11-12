"""개방셋(open-set) 유틸 테스트 — 클래스 서브셋·분할·평가 마스크 (plan4.md).

torchvision 다운로드 없이 최소 페이크 데이터셋으로 검증(CPU, 수 초).
"""

import pickle

import numpy as np

from src.datasets.image import ClassSubsetDataset, select_seen_classes


class _FakeDS:
    """targets/__getitem__만 가진 최소 데이터셋 (torchvision 다운로드 회피)."""

    def __init__(self, targets):
        self.targets = list(targets)
        self.data = np.arange(len(targets))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.data[i], self.targets[i]


def test_select_seen_classes_deterministic():
    a = select_seen_classes(100, 80, seed=0)
    b = select_seen_classes(100, 80, seed=0)
    assert np.array_equal(a, b)                       # 동일 시드 = 결정적
    assert len(a) == 80 and len(set(a.tolist())) == 80  # 중복 없음
    assert np.array_equal(a, np.sort(a))              # 정렬됨
    assert a.min() >= 0 and a.max() < 100
    assert not np.array_equal(a, select_seen_classes(100, 80, seed=1))  # 시드 다르면 분할도 다름


def test_select_seen_complement():
    seen = select_seen_classes(100, 80, seed=0)
    novel = np.setdiff1d(np.arange(100), seen)
    assert len(novel) == 20
    assert len(np.intersect1d(seen, novel)) == 0      # seen ∩ novel = ∅


def test_class_subset_keeps_only_seen_and_original_ids():
    targets = [i % 10 for i in range(100)]            # 클래스 0..9, 각 10개
    base = _FakeDS(targets)
    keep = [1, 3, 7]
    sub = ClassSubsetDataset(base, keep)
    assert set(np.asarray(sub.targets).tolist()) == set(keep)  # seen 클래스만
    assert len(sub) == 30                             # 3클래스 × 10샘플
    for i in range(len(sub)):
        _, y = sub[i]
        assert int(y) in keep                         # 원본 id 유지(리맵 없음)
    assert hasattr(sub, "targets")                    # get_labels/PKBatchSampler 호환


def test_class_subset_picklable():
    """Windows spawn 대비 — 모듈레벨 클래스는 피클 가능해야."""
    base = _FakeDS([i % 5 for i in range(50)])
    sub = ClassSubsetDataset(base, [0, 2])
    restored = pickle.loads(pickle.dumps(sub))
    assert len(restored) == len(sub) == 20


def test_openset_eval_masks_and_top1():
    """seen/novel 마스크 분리 + predict 기반 top1 산출 정합성(토이)."""
    y = np.array([1, 3, 7, 50, 90])                   # 앞 3개 seen, 뒤 2개 novel
    seen = np.array([1, 3, 7])
    seen_mask = np.isin(y, seen)
    novel_mask = ~seen_mask
    assert seen_mask.sum() == 3 and novel_mask.sum() == 2
    pred = np.array([1, 3, 0, 0, 0])                  # seen 2/3 맞음, novel 0/2
    correct = pred == y
    assert np.isclose(correct.mean(), 2 / 5)          # open_top1_all
    assert np.isclose(correct[seen_mask].mean(), 2 / 3)   # open_top1_seen
    assert np.isclose(correct[novel_mask].mean(), 0.0)    # open_top1_novel
