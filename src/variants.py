"""v3 변형(variant) 레지스트리 — config 4축(energy/logit/apt/bank) 배관.

설계 규칙: config 배관이므로 src/nmfc(순수 수학) 밖에 둔다. 핵심 불변식:
- `variant:` 블록이 없으면 = v1 = 바이트 동일(resolve가 V1_DEFAULTS, build가 기존 NMFCHead).
- is_v1은 energy/logit/bank 3축만 본다(apt 축 직교 — mmjb는 v1 헤드 + ACS 워밍업).
- 무효 조합은 학습 시작 전에 한국어 ValueError로 거부(침묵 실패/클램프 금지).
"""

from src.models import NMFCHead, VariantNMFCHead

V1_DEFAULTS = {"energy": "mean", "logit": "mean_scaled", "apt": "reciprocal", "bank": "minibatch"}

_AXIS_VALUES = {
    "energy": {"mean", "softmin"},
    "logit": {"mean_scaled", "hinge_temp"},
    "apt": {"reciprocal", "negJ", "negJ_sphere", "negJ_trace"},
    "bank": {"minibatch", "online"},
}

_CODE = {
    "energy": {"mean": "m", "softmin": "s"},
    "logit": {"mean_scaled": "m", "hinge_temp": "h"},
    "apt": {"reciprocal": "r", "negJ": "j", "negJ_sphere": "n", "negJ_trace": "t"},
    "bank": {"minibatch": "b", "online": "o"},
}

# negJ 계열(전부 ACSController -J 워밍업). v4: sphere/trace는 -J의 산포/포화만 다름.
_NEGJ_FAMILY = ("negJ", "negJ_sphere", "negJ_trace")


def is_negj_family(apt: str) -> bool:
    """apt가 -J 워밍업 계열(negJ/negJ_sphere/negJ_trace)이면 True. 침묵 실패 방지용 중앙화."""
    return apt in _NEGJ_FAMILY


def resolve_variant(spec: dict) -> dict:
    """spec의 variant 블록을 V1_DEFAULTS 위에 덮어 4축 완전 명세를 만든다."""
    return {**V1_DEFAULTS, **spec.get("variant", {})}


def is_v1(variant: dict) -> bool:
    """energy/logit/bank 3축이 모두 v1이면 True (apt 축은 무관)."""
    return all(variant[ax] == V1_DEFAULTS[ax] for ax in ("energy", "logit", "bank"))


def variant_code(variant: dict) -> str:
    """[energy, logit, apt, bank] → 4자 코드. v1=mmrb, 완전 v3=shjo."""
    return "".join(_CODE[ax][variant[ax]] for ax in ("energy", "logit", "apt", "bank"))


def validate_variant(variant: dict, spec: dict, cfg: dict | None = None) -> None:
    """조합 사전 검증. 무효면 한국어 ValueError. method!=nmfc면 no-op(단, variant 블록은 거부)."""
    if spec.get("method") != "nmfc":
        if spec.get("variant"):
            raise ValueError("variant: 블록은 method: nmfc 실험에만 허용됩니다 (다른 method는 4축 무시).")
        return
    for ax, allowed in _AXIS_VALUES.items():
        if variant[ax] not in allowed:
            raise ValueError(f"variant.{ax}={variant[ax]!r}는 무효입니다 (허용: {sorted(allowed)}).")
    if variant["energy"] == "softmin" and variant["logit"] != "hinge_temp":
        raise ValueError(
            "energy: softmin은 logit: hinge_temp가 필수입니다 "
            "(soft-min E는 음수 가능 → μ± 나눗셈이 부호 붕괴)."
        )
    if variant["bank"] == "online" and variant["energy"] != "softmin":
        raise ValueError("bank: online은 energy: softmin이 필수입니다 (kNN 이웃집합은 뱅크를 전제).")
    if variant["bank"] == "minibatch" and variant["energy"] == "softmin":
        nm = {**(cfg or {}).get("nmfc", {}), **spec.get("nmfc", {})}
        k1, k2 = nm["k1"], nm["k2"]
        bp, bk = spec["batch_p"], spec["batch_k"]
        if k1 > bk - 1:
            raise ValueError(
                f"배치 내 softmin k1={k1} > batch_k−1={bk - 1} — PK LOO 후 남는 동일 클래스 이웃 상한 "
                f"초과 (침묵 클램프 금지; nmfc.k1을 낮추거나 batch_k를 키우세요)."
            )
        if k2 > (bp - 1) * bk:
            raise ValueError(
                f"배치 내 softmin k2={k2} > (batch_p−1)·batch_k={(bp - 1) * bk} — 타 클래스 이웃 상한 초과."
            )


def build_nmfc_head(spec: dict, cfg: dict, variant: dict):
    """is_v1이면 기존 NMFCHead(객체 동일, 캐시 안전), 아니면 VariantNMFCHead.

    v1 분기는 train.py:make_nmfc_head와 동일한 인자로 NMFCHead를 만든다(바이트 동일).
    """
    nm = {**cfg["nmfc"], **spec.get("nmfc", {})}
    if is_v1(variant):
        return NMFCHead(lam=nm["lam"], n_ref=nm["n_ref"], detach_pi=nm["detach_pi"])
    return VariantNMFCHead(
        energy=variant["energy"],
        logit=variant["logit"],
        lam=nm["lam"],
        k1=nm["k1"],
        k2=nm["k2"],
        tau_min=nm["tau_min"],
        m_quantile=nm["m_quantile"],
        n_coreset=nm["n_coreset"],
        n_ref=nm["n_ref"],
        detach_pi=nm["detach_pi"],
        bank=variant["bank"],
    )


def build_warmup(spec: dict, cfg: dict, variant: dict):
    """apt이 negJ 계열이면 ACSController(cfg['acs']+spec['acs']), 아니면 기존 make_apt(reciprocal)."""
    if is_negj_family(variant["apt"]):
        from src.nmfc.acs import ACSController

        a = {**cfg.get("acs", {}), **spec.get("acs", {})}
        return ACSController(
            window=a["window"],
            patience=a["patience"],
            delta=a["delta"],
            t_min=a["t_min"],
            t_max=a["t_max"],
            ramp=a["ramp"],
            eps_phi=a.get("eps_phi", 1e-12),
        )
    from src.train import make_apt  # 지연 import(순환 방지) — reciprocal은 v1 APT 경로 재사용

    return make_apt(spec, cfg)
