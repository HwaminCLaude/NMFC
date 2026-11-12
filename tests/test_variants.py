import pytest

from src.models import NMFCHead, VariantNMFCHead
from src.nmfc.acs import ACSController
from src.variants import (
    V1_DEFAULTS,
    build_nmfc_head,
    build_warmup,
    is_negj_family,
    is_v1,
    resolve_variant,
    validate_variant,
    variant_code,
)


def _cfg():
    return {
        "nmfc": {"lam": 0.5, "n_ref": 64, "detach_pi": False, "k1": 16, "k2": 16,
                 "m_quantile": 0.75, "tau_min": 0.01, "bank_q": 64, "n_coreset": 64},
        "apt": {"window": 5, "delta": 1e-4, "patience": 5, "min_warmup": 5, "max_warmup": 30,
                "probe_size": 5000},
        "acs": {"window": 5, "patience": 5, "delta": 1e-4, "t_min": 10, "t_max": 30, "ramp": 5,
                "eps_phi": 1e-12},
    }


def test_resolve_default_is_v1():
    v = resolve_variant({})
    assert v == V1_DEFAULTS
    assert is_v1(v)


def test_build_v1_returns_plain_nmfc_head():
    spec = {"method": "nmfc", "name": "x"}
    head = build_nmfc_head(spec, _cfg(), resolve_variant(spec))
    assert isinstance(head, NMFCHead) and not isinstance(head, VariantNMFCHead)


def test_build_variant_returns_variant_head():
    spec = {"method": "nmfc", "variant": {"energy": "softmin", "logit": "hinge_temp"},
            "nmfc": {"k1": 3, "k2": 32}}
    head = build_nmfc_head(spec, _cfg(), resolve_variant(spec))
    assert isinstance(head, VariantNMFCHead)
    assert head.energy == "softmin" and head.k1 == 3 and head.k2 == 32


def test_is_v1_ignores_apt_axis():
    # mmjb: apt만 negJ, 나머지 v1 → is_v1 True → v1 헤드
    v = resolve_variant({"variant": {"apt": "negJ"}})
    assert is_v1(v)
    assert isinstance(build_nmfc_head({"method": "nmfc"}, _cfg(), v), NMFCHead)


def test_variant_code_roundtrip():
    assert variant_code(V1_DEFAULTS) == "mmrb"
    assert variant_code({"energy": "softmin", "logit": "hinge_temp", "apt": "negJ",
                         "bank": "online"}) == "shjo"
    assert variant_code(resolve_variant({"variant": {"logit": "hinge_temp"}})) == "mhrb"


def test_invalid_softmin_needs_hinge():
    spec = {"method": "nmfc", "variant": {"energy": "softmin"}}  # logit=mean_scaled 유지
    with pytest.raises(ValueError):
        validate_variant(resolve_variant(spec), spec, _cfg())


def test_invalid_online_needs_softmin():
    spec = {"method": "nmfc", "variant": {"bank": "online", "logit": "hinge_temp"}}  # energy=mean
    with pytest.raises(ValueError):
        validate_variant(resolve_variant(spec), spec, _cfg())


def test_invalid_k1_exceeds_batch():
    spec = {"method": "nmfc", "batch_p": 48, "batch_k": 4,
            "variant": {"energy": "softmin", "logit": "hinge_temp"}, "nmfc": {"k1": 5, "k2": 32}}
    with pytest.raises(ValueError):  # 5 > batch_k−1=3
        validate_variant(resolve_variant(spec), spec, _cfg())


def test_variant_block_rejected_on_non_nmfc():
    spec = {"method": "softmax", "variant": {"energy": "softmin"}}
    with pytest.raises(ValueError):
        validate_variant(resolve_variant(spec), spec, _cfg())


def test_valid_combo_passes():
    spec = {"method": "nmfc", "batch_p": 48, "batch_k": 4,
            "variant": {"energy": "softmin", "logit": "hinge_temp"}, "nmfc": {"k1": 3, "k2": 32}}
    validate_variant(resolve_variant(spec), spec, _cfg())  # 예외 없음


def test_build_warmup_negj_returns_acs():
    spec = {"method": "nmfc", "variant": {"energy": "softmin", "logit": "hinge_temp", "apt": "negJ"}}
    w = build_warmup(spec, _cfg(), resolve_variant(spec))
    assert isinstance(w, ACSController)


# ---- v4: apt 축 sphere/trace 확장 ----


def test_is_negj_family():
    assert is_negj_family("negJ")
    assert is_negj_family("negJ_sphere")
    assert is_negj_family("negJ_trace")
    assert not is_negj_family("reciprocal")


@pytest.mark.parametrize("apt,code", [("negJ_sphere", "shnb"), ("negJ_trace", "shtb")])
def test_variant_code_v4_apt(apt, code):
    """v4 apt 코드: negJ_sphere→n, negJ_trace→t (positional [energy,logit,apt,bank])."""
    v = resolve_variant({"variant": {"energy": "softmin", "logit": "hinge_temp", "apt": apt}})
    assert variant_code(v) == code


@pytest.mark.parametrize("apt", ["negJ_sphere", "negJ_trace"])
def test_build_warmup_v4_apt_returns_acs(apt):
    spec = {"method": "nmfc", "variant": {"energy": "softmin", "logit": "hinge_temp", "apt": apt}}
    w = build_warmup(spec, _cfg(), resolve_variant(spec))
    assert isinstance(w, ACSController)


@pytest.mark.parametrize("apt", ["negJ_sphere", "negJ_trace"])
def test_v4_apt_validates_and_is_v1_ignores_apt(apt):
    # sphere/trace apt는 유효값이고, is_v1은 apt 축을 무시하므로 v1 헤드(mm?b)여도 True
    spec = {"method": "nmfc", "batch_p": 48, "batch_k": 4,
            "variant": {"energy": "softmin", "logit": "hinge_temp", "apt": apt},
            "nmfc": {"k1": 3, "k2": 32}}
    validate_variant(resolve_variant(spec), spec, _cfg())  # 예외 없음
    assert is_v1(resolve_variant({"variant": {"apt": apt}}))  # apt만 바뀌면 여전히 v1 헤드
