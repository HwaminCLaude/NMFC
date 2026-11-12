from src.nmfc import APTController, FixedTransition, NoWarmup
from src.nmfc.apt import LINEAR, NONLINEAR


def test_fires_on_plateau():
    apt = APTController(window=5, delta=1e-4, patience=5, min_warmup=5, max_warmup=30)
    seq = [0.1 * t for t in range(1, 11)] + [1.0] * 20  # 상승 후 plateau
    fired_at = None
    for r in seq:
        if apt.update(r) == NONLINEAR and fired_at is None:
            fired_at = len(apt.ratios)
    assert fired_at is not None
    # plateau 시작(에폭 11) 후 window+patience 이내에 발동해야 함
    assert 11 <= fired_at <= 11 + apt.window + apt.patience


def test_does_not_fire_while_rising():
    apt = APTController(window=5, delta=1e-4, patience=5, min_warmup=5, max_warmup=30)
    for t in range(1, 30):  # max_warmup 못 미치는 동안 계속 상승
        phase = apt.update(0.5 * t)
    assert phase == LINEAR


def test_max_warmup_cap_fires():
    apt = APTController(window=5, delta=1e-4, patience=5, min_warmup=5, max_warmup=30)
    for t in range(1, 40):
        apt.update(0.5 * t)  # 영원히 상승해도
    assert apt.phase == NONLINEAR
    assert apt.transition_epoch == 30
    assert apt.transition_reason == "max_warmup"


def test_one_way_transition():
    apt = APTController(min_warmup=2, patience=2, max_warmup=30)
    for r in [1.0, 1.0, 1.0, 1.0]:
        apt.update(r)
    assert apt.phase == NONLINEAR
    for r in [10.0, 100.0]:  # 다시 크게 변해도 역전환 없음
        assert apt.update(r) == NONLINEAR


def test_ablation_controllers():
    fixed = FixedTransition(fixed_epoch=3)
    phases = [fixed.update(float(t)) for t in range(1, 6)]
    assert phases == [LINEAR, LINEAR, NONLINEAR, NONLINEAR, NONLINEAR]
    assert fixed.transition_epoch == 3

    none = NoWarmup()
    assert none.update(1.0) == NONLINEAR
    assert none.transition_epoch == 0
