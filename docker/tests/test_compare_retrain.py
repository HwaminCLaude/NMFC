"""compare(재학습 결과) 판정 테스트 — t_factor, retrain_compare(전제조건·신선도·단일 런·평균·부호·정확 일치).

기준본은 저장소 results/의 json·md·csv 사본, '현재'는 그 사본을 고친 가짜 재학습 결과다.
mtime은 os.utime으로 직접 정해 신선도 판정을 결정적으로 만든다.
"""

import ast
import hashlib
import json
import math
import os
import shutil
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest
import yaml

DOCKER = Path(__file__).resolve().parents[1]
REPO = DOCKER.parent
if str(DOCKER) not in sys.path:
    sys.path.insert(0, str(DOCKER))

import nmfc_compare as C  # noqa: E402

TOL = C.load_yaml(DOCKER / "tolerances.yaml")
RT = TOL["retrain"]
CORE_METRICS = ("top1", "silhouette", "map", "fewshot_s1", "verif_auc")
K_SINGLE = None  # t_factor(0.05, 5, 4)·√2 — 테스트 안에서 계산


# --------------------------------------------------------------------------- 통계 계수·매핑

def test_t_factor_values():
    assert C.t_factor(0.05, 5, 4) == pytest.approx(4.6041, abs=1e-4)
    assert C.t_factor(0.05, 5, 4) * math.sqrt(2) == pytest.approx(6.511, abs=1e-3)       # 단일 런 m_run=5
    assert C.t_factor(0.05, 20, 4) * math.sqrt(2 / 5) == pytest.approx(4.274, abs=1e-3)  # 코어 평균 n=5,m=20
    assert C.t_factor(0.05, 36, 3) * math.sqrt(2 / 4) == pytest.approx(8.176, abs=1e-3)  # 그리드 평균 n=4,m=36
    assert C.t_factor(0.05, 40, 4) > C.t_factor(0.05, 20, 4)                                # m이 크면 문턱도 크다


def test_retrain_parameters_are_as_registered():
    assert RT["alpha"] == 0.05 and RT["single_run"]["df"] == 4
    assert (RT["families"]["core"]["n"], RT["families"]["core"]["m"]) == (5, 20)
    assert (RT["families"]["grid"]["n"], RT["families"]["grid"]["m"]) == (4, 36)
    assert RT["synthetic"]["metrics"] == {"test_top1": 0.01, "silhouette": 0.02}
    assert RT["single_seed"]["report_abs"] == 0.02 and RT["report_sd_multiple"] == 4.2
    codes = RT["transition_epoch"]["report_variant_codes"]
    assert codes == ["shjb", "shjo"]
    committed = {json.loads(p.read_text(encoding="utf-8")).get("variant_code")
                 for p in (REPO / "results").glob("*/metrics.json")}
    assert set(codes) <= committed                                # 커밋된 런의 variant_code만 둔다
    assert [(g["a"], g["b"], g["metrics"]) for g in RT["sign_gate"]["gate"]] == [
        ("shrb", "softmax", ["map", "fewshot_s1", "silhouette"]), ("shrb", "mhrb", ["map", "silhouette"])]
    assert len(RT["exact_keys"]) == 9


def test_family_mapping_matches_eval_seed_robustness_and_configs():
    tree = ast.parse((REPO / "scripts" / "eval_seed_robustness.py").read_text(encoding="utf-8"))
    consts = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ("SEEDS", "CORE", "OOD", "DML")):
            consts[node.targets[0].id] = ast.literal_eval(node.value)
    fam = RT["families"]
    assert fam["core"]["seeds"] == consts["SEEDS"] == fam["ood"]["seeds"]
    assert fam["core"]["roles"] == consts["CORE"]
    assert fam["ood"]["roles"] == consts["OOD"]
    assert fam["dml"]["roles"] == consts["DML"]
    names = set()
    for p in (REPO / "configs").glob("*seeds_s*.yaml"):
        names |= {e["name"] for e in yaml.safe_load(p.read_text(encoding="utf-8"))["experiments"]}
    for fname in ("core", "ood"):
        for base in fam[fname]["roles"].values():
            for s in fam[fname]["seeds"]:
                if s != fam[fname]["base_seed"]:
                    assert f"{base}_s{s}" in names
    grid = json.loads((REPO / "results" / "openset_noise_grid" / "master.json").read_text(encoding="utf-8"))
    cells = {}
    for r in grid["rows"]:
        cells.setdefault((r["method"], r["scope"], r["rate"]), set()).add(r["seed"])
    assert len(cells) == 18 and all(len(v) == 4 for v in cells.values())
    assert len(cells) * len(fam["grid"]["metrics"]) == fam["grid"]["m"]


# --------------------------------------------------------------------------- 가짜 기준본·현재 트리

@pytest.fixture(scope="module")
def base_tree(tmp_path_factory):
    root = tmp_path_factory.mktemp("rt_base")
    for p in (REPO / "results").rglob("*"):
        if p.is_file() and p.suffix in (".json", ".md", ".csv"):
            dst = root / p.relative_to(REPO)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dst)
    return root


@pytest.fixture
def env(base_tree, tmp_path, monkeypatch):
    # 컨테이너 안(NMFC_IN_CONTAINER=1)에서도 임시 트리를 판정할 수 있게 볼륨 검사를 끈다(검사 자체는 별도 테스트).
    monkeypatch.delenv("NMFC_IN_CONTAINER", raising=False)
    cur = tmp_path / "cur"
    shutil.copytree(base_tree, cur)
    t0 = time.time() - 100_000
    for p in cur.rglob("*"):
        if p.is_file():
            os.utime(p, (t0, t0))
    return base_tree, cur, t0


def _set_mtime(p: Path, t: float):
    os.utime(p, (t, t))


def retrain(cur: Path, exp: str, t: float, **changes):
    """exp를 다시 학습한 것처럼: embeddings.npz를 t에, 바꾼 metrics.json을 t+1에 쓴다."""
    d = cur / "results" / exp
    (d / "embeddings.npz").write_bytes(b"fake npz " + exp.encode())
    _set_mtime(d / "embeddings.npz", t)
    m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
    for k, v in changes.items():
        path = k.split("__")
        tgt = m
        for p in path[:-1]:
            tgt = tgt[p]
        tgt[path[-1]] = v(tgt[path[-1]]) if callable(v) else v
    (d / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    _set_mtime(d / "metrics.json", t + 1)


def aggregate(cur: Path, t: float, mutate=None, rel="results/seed_robustness.json"):
    p = cur / rel
    d = json.loads(p.read_text(encoding="utf-8"))
    if mutate:
        mutate(d)
    p.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    _set_mtime(p, t)


def artifacts(cur: Path, trained) -> dict:
    """nmfc_pipeline.run_artifacts와 같은 형식의 학습 산출물 지문(학습 직후 = 지금 파일)."""
    out = {}
    for e in trained:
        d = cur / "results" / e
        fp = {}
        for fn in ("metrics.json", "embeddings.npz"):
            p = d / fn
            if p.is_file():
                b = p.read_bytes()
                fp[fn] = {"sha256": hashlib.sha256(b).hexdigest(), "size": len(b), "mtime_ns": p.stat().st_mtime_ns}
            else:
                fp[fn] = None
        if fp["embeddings.npz"] is not None:
            fp["embeddings.npz"]["rewritten"] = True
        if fp["metrics.json"] is not None:
            out[e] = fp
    return out


def status(t0, trained, cur=None):
    """파이프라인 상태 흉내. cur를 주면 학습 항목 하나에 산출물 지문(nmfc_pipeline이 학습 직후 남기는 것)을 단다."""
    items = []
    if cur is not None:
        items = [{"id": "T.test", "kind": "train", "required": True, "exit": 0, "ok": True,
                  "expected": list(trained), "cached": [], "trained": list(trained),
                  "artifacts": artifacts(cur, trained)}]
    return {"started": datetime.fromtimestamp(t0 + 100).isoformat(), "finished": None,
            "args": {}, "items": items, "trained": list(trained), "pass": True}


def seed_sd(role, metric):
    sr = json.loads((REPO / "results" / "seed_robustness.json").read_text(encoding="utf-8"))
    return sr["core"][role][metric]["sd"]


def k_single():
    return C.t_factor(0.05, 5, 4) * math.sqrt(2)


def bump_core(d, role, seed, frac, metrics=("map", "fewshot_s1", "verif_auc"), k=None):
    k = k if k is not None else k_single()
    for m in metrics:
        d["core"][role][m]["by_seed"][str(seed)] += frac * k * d["core"][role][m]["sd"]


def gates(r, kind=None, status_=None):
    return [g for g in r["gates"] if (kind is None or g["kind"] == kind) and (status_ is None or g["status"] == status_)]


# --------------------------------------------------------------------------- 단일 런 게이트

def test_single_run_gate_pass(env):
    base, cur, t0 = env
    k = k_single()
    retrain(cur, "c100v_shrb", t0 + 200, top1=lambda v: v + 0.5 * k * seed_sd("shrb", "top1"),
            silhouette=lambda v: v - 0.5 * k * seed_sd("shrb", "silhouette"))
    aggregate(cur, t0 + 300, lambda d: bump_core(d, "shrb", 42, 0.5))
    r = C.retrain_compare(base, cur, status(t0, ["c100v_shrb"], cur), TOL)
    assert r["precondition"]["ok"] and r["unevaluated"] == []
    assert r["exit_code"] == 0 and r["pass"] is True, [g for g in r["gates"] if g["status"] == "FAIL"]
    single = gates(r, "single_run")
    assert sorted(g["metric"] for g in single) == sorted(CORE_METRICS)
    for g in single:
        assert g["status"] == "PASS" and g["m_run"] == 5 and g["df"] == 4
        assert g["s"] == pytest.approx(seed_sd("shrb", g["metric"]), rel=1e-9)   # 기준본 5시드 표본 SD
        assert g["threshold"] == pytest.approx(k * g["s"], rel=1e-9)
        assert g["delta"] == pytest.approx(0.5 * g["threshold"], rel=1e-6)
    assert sorted(g["group"] for g in gates(r, "sign")) == sorted([
        "shrb_vs_softmax/map", "shrb_vs_softmax/fewshot_s1", "shrb_vs_softmax/silhouette",
        "shrb_vs_mhrb/map", "shrb_vs_mhrb/silhouette"])
    assert all(g["status"] == "PASS" and g["new"] == "5/5" for g in gates(r, "sign"))
    te = [g for g in gates(r, "exact") if g["metric"] == "transition_epoch"]
    assert len(te) == 1 and te[0]["status"] == "PASS"
    assert not gates(r, "mean")                                   # 5시드 전부가 아니므로 평균 게이트 없음
    cats = {x["path"]: x["category"] for x in r["reports"] if x.get("kind") == "file_change"}
    assert cats["results/c100v_shrb/metrics.json"] == "trained"
    assert cats["results/seed_robustness.json"] == "aggregate"


@pytest.mark.parametrize("metric", ["top1", "silhouette"])
def test_single_run_gate_exceeded_fails(env, metric):
    base, cur, t0 = env
    retrain(cur, "c100v_shrb", t0 + 200, **{metric: lambda v: v + 1.2 * k_single() * seed_sd("shrb", metric)})
    aggregate(cur, t0 + 300, lambda d: bump_core(d, "shrb", 42, 0.1))
    r = C.retrain_compare(base, cur, status(t0, ["c100v_shrb"], cur), TOL)
    assert r["exit_code"] == 1 and r["pass"] is False
    assert [g["metric"] for g in gates(r, "single_run", "FAIL")] == [metric]


def test_aggregate_metric_exceeded_fails(env):
    base, cur, t0 = env
    retrain(cur, "c100v_shrb", t0 + 200)
    aggregate(cur, t0 + 300, lambda d: bump_core(d, "shrb", 42, 1.3, metrics=("map",)))
    r = C.retrain_compare(base, cur, status(t0, ["c100v_shrb"], cur), TOL)
    assert r["exit_code"] == 1
    assert [g["metric"] for g in gates(r, "single_run", "FAIL")] == ["map"]


# --------------------------------------------------------------------------- 신선도·전제조건

def test_stale_aggregate_is_unevaluated(env):
    base, cur, t0 = env
    retrain(cur, "c100v_shrb", t0 + 200)                       # 집계를 다시 돌리지 않음
    r = C.retrain_compare(base, cur, status(t0, ["c100v_shrb"], cur), TOL)
    assert r["precondition"]["ok"]
    stale = {u["metric"] for u in r["unevaluated"] if u.get("exp") == "c100v_shrb"}
    assert stale == {"map", "fewshot_s1", "verif_auc"}
    assert all("오래됨" in u["reason"] for u in r["unevaluated"] if u.get("exp") == "c100v_shrb")
    assert {u["group"] for u in r["unevaluated"] if u.get("gate") == "sign"} == {
        "shrb_vs_softmax/map", "shrb_vs_softmax/fewshot_s1", "shrb_vs_mhrb/map"}
    assert {g["metric"] for g in gates(r, "single_run", "PASS")} == {"top1", "silhouette"}
    assert r["exit_code"] == 2 and r["pass"] is False


def test_aggregate_identical_to_baseline_is_unevaluated(env):
    base, cur, t0 = env
    retrain(cur, "c100v_shrb", t0 + 200)
    aggregate(cur, t0 + 300)                                   # 새 mtime, 내용은 기준본과 같음
    r = C.retrain_compare(base, cur, status(t0, ["c100v_shrb"], cur), TOL)
    reasons = {u["reason"] for u in r["unevaluated"] if u.get("exp") == "c100v_shrb"}
    assert reasons and all("기준본과 내용이 같음" in x for x in reasons)
    assert r["exit_code"] == 2


def test_no_training_is_precondition_failure(env):
    base, cur, t0 = env
    for st in (status(t0, [], cur), {}, None):
        r = C.retrain_compare(base, cur, st, TOL)
        assert r["exit_code"] == 2 and r["pass"] is False and r["precondition"]["ok"] is False
        assert r["gates"] == []


def test_metrics_older_than_start_is_precondition_failure(env):
    base, cur, t0 = env
    retrain(cur, "c100v_shrb", t0 + 200)
    st = status(t0, ["c100v_shrb"], cur)
    st["started"] = datetime.fromtimestamp(t0 + 500).isoformat()
    r = C.retrain_compare(base, cur, st, TOL)
    assert r["exit_code"] == 2 and any("실행 시작보다 이전" in x for x in r["precondition"]["reasons"])
    st2 = status(t0, ["no_such_exp"], cur)
    r = C.retrain_compare(base, cur, st2, TOL)
    assert r["exit_code"] == 2 and any("metrics.json 없음" in x for x in r["precondition"]["reasons"])


def test_item_level_start_and_pipeline_iso_format(env):
    base, cur, t0 = env
    retrain(cur, "c100v_shrb", t0 + 200)                       # metrics.json mtime = t0+201
    aggregate(cur, t0 + 300, lambda d: bump_core(d, "shrb", 42, 0.1))
    st = status(t0, ["c100v_shrb"], cur)
    st["started"] = datetime.fromtimestamp(t0 + 100).astimezone().isoformat(timespec="milliseconds")
    assert C.retrain_compare(base, cur, st, TOL)["exit_code"] == 0
    st["items"] = [{"id": "T", "kind": "train", "exit": 0, "ok": True, "trained": ["c100v_shrb"],
                    "started_ns": int((t0 + 250) * 1e9), "artifacts": artifacts(cur, ["c100v_shrb"])}]
    r = C.retrain_compare(base, cur, st, TOL)                 # 항목 시작(t0+250)보다 metrics가 오래됨
    assert r["exit_code"] == 2 and r["precondition"]["ok"] is False
    st["items"][0]["started_ns"] = int((t0 + 150) * 1e9)
    assert C.retrain_compare(base, cur, st, TOL)["exit_code"] == 0


def test_container_requires_results_volume(env, monkeypatch):
    base, cur, t0 = env
    retrain(cur, "c100v_shrb", t0 + 200)
    aggregate(cur, t0 + 300, lambda d: bump_core(d, "shrb", 42, 0.1))
    monkeypatch.setenv("NMFC_IN_CONTAINER", "1")
    r = C.retrain_compare(base, cur, status(t0, ["c100v_shrb"], cur), TOL)
    assert r["exit_code"] == 2 and any("볼륨" in x for x in r["precondition"]["reasons"])
    monkeypatch.delenv("NMFC_IN_CONTAINER")
    assert C.retrain_compare(base, cur, status(t0, ["c100v_shrb"], cur), TOL)["exit_code"] == 0


def test_only_report_only_runs_is_not_a_pass(env):
    base, cur, t0 = env
    retrain(cur, "c100v_shjb", t0 + 200)                       # 단일 시드 변형: 게이트 0건
    r = C.retrain_compare(base, cur, status(t0, ["c100v_shjb"], cur), TOL)
    assert r["gates"] == [] and r["exit_code"] == 2 and r["pass"] is False


# --------------------------------------------------------------------------- 정확 일치·transition_epoch·합성

def test_openset_grid_run_exact_items_and_grid_gate(env):
    base, cur, t0 = env
    retrain(cur, "c100v_shrb_open_na20", t0 + 200)
    r = C.retrain_compare(base, cur, status(t0, ["c100v_shrb_open_na20"], cur), TOL)
    assert r["exit_code"] == 0, r["unevaluated"] or gates(r, status_="FAIL")
    exact = {g["metric"] for g in gates(r, "exact")}
    assert set(RT["exact_keys"]) | {"transition_epoch"} == exact
    grid = gates(r, "single_run")
    assert sorted(g["metric"] for g in grid) == ["closed_seen_top1", "open_ood_auc"]
    rows = json.loads((REPO / "results" / "openset_noise_grid" / "master.json").read_text(encoding="utf-8"))["rows"]
    cell = [x["open_ood_auc"] for x in rows if (x["method"], x["scope"], x["rate"]) == ("shrb", "na", 20)]
    g = next(g for g in grid if g["metric"] == "open_ood_auc")
    assert g["group"] == "shrb/na/20" and g["m_run"] == 2
    assert g["s"] == pytest.approx(statistics.stdev(cell), rel=1e-12)    # 셀별 4시드 SD


@pytest.mark.parametrize("key,value", [
    ("label_noise_stats__n_flipped", 9999), ("label_noise_stats__train_size", 40000),
    ("label_noise__seed", 1), ("split_seed", 1), ("n_seen", 79),
    ("seen_classes", lambda v: v[:-1] + [100])])
def test_exact_item_change_fails(env, key, value):
    base, cur, t0 = env
    retrain(cur, "c100v_shrb_open_na20", t0 + 200, **{key: value})
    r = C.retrain_compare(base, cur, status(t0, ["c100v_shrb_open_na20"], cur), TOL)
    assert r["exit_code"] == 1
    assert [g["metric"] for g in gates(r, "exact", "FAIL")] == [key.replace("__", ".")]


def test_transition_epoch_reciprocal_exact_negj_report(env):
    base, cur, t0 = env
    retrain(cur, "c100v_shrb", t0 + 200, transition_epoch=6)
    aggregate(cur, t0 + 300, lambda d: bump_core(d, "shrb", 42, 0.1))
    r = C.retrain_compare(base, cur, status(t0, ["c100v_shrb"], cur), TOL)
    assert r["exit_code"] == 1 and [g["metric"] for g in gates(r, "exact", "FAIL")] == ["transition_epoch"]

    retrain(cur, "c100v_shrb", t0 + 200, transition_epoch=5)   # 정상으로 되돌림(같은 mtime)
    retrain(cur, "c100v_shjb", t0 + 210, transition_epoch=7)
    r = C.retrain_compare(base, cur, status(t0, ["c100v_shrb", "c100v_shjb"], cur), TOL)
    assert r["exit_code"] == 0, gates(r, status_="FAIL")
    rep = [x for x in r["reports"] if x.get("kind") == "transition_epoch" and x["exp"] == "c100v_shjb"]
    assert len(rep) == 1 and rep[0]["ref"] == 8 and rep[0]["new"] == 7
    na = [x for x in r["reports"] if x.get("exp") == "c100v_shjb" and x["status"] == "N/A"]
    assert {x["metric"] for x in na} == {"mAP", "recall@1"}    # retrieval_eval 미갱신 → 보고 N/A(게이트 아님)


@pytest.mark.parametrize("delta,metric,code", [(0.005, "test_top1", 0), (0.02, "test_top1", 1),
                                                (0.015, "silhouette", 0), (0.03, "silhouette", 1)])
def test_synthetic_preregistered_tolerance(env, delta, metric, code):
    base, cur, t0 = env
    retrain(cur, "syn_moons_nmfc", t0 + 200, **{metric: lambda v: v + delta})
    r = C.retrain_compare(base, cur, status(t0, ["syn_moons_nmfc"], cur), TOL)
    assert r["exit_code"] == code
    syn = gates(r, "synthetic")
    assert sorted(g["metric"] for g in syn) == ["silhouette", "test_top1"]
    assert any(x.get("kind") == "transition_epoch" for x in r["reports"])     # apt 없음 → 보고만


# --------------------------------------------------------------------------- 5시드 평균 게이트·부호 게이트

SHRB = ["c100v_shrb", "c100v_shrb_s0", "c100v_shrb_s1", "c100v_shrb_s2", "c100v_shrb_s3"]


def _retrain_all_shrb(cur, t0, frac_sd):
    for i, exp in enumerate(SHRB):
        retrain(cur, exp, t0 + 200 + i, top1=lambda v: v + frac_sd * seed_sd("shrb", "top1"))

    def mut(d):
        for s in (42, 0, 1, 2, 3):
            for m in ("map", "fewshot_s1", "verif_auc"):
                d["core"]["shrb"][m]["by_seed"][str(s)] += 0.3 * d["core"]["shrb"][m]["sd"]
    aggregate(cur, t0 + 400, mut)


def test_mean_gate_used_when_all_five_seeds_retrained(env):
    base, cur, t0 = env
    _retrain_all_shrb(cur, t0, frac_sd=1.0)
    r = C.retrain_compare(base, cur, status(t0, SHRB, cur), TOL)
    assert r["exit_code"] == 0, gates(r, status_="FAIL") or r["unevaluated"]
    means = gates(r, "mean")
    assert sorted(g["metric"] for g in means) == sorted(CORE_METRICS)
    k_mean = C.t_factor(0.05, 20, 4) * math.sqrt(2 / 5)
    for g in means:
        assert g["n"] == 5 and g["m"] == 20 and g["threshold"] == pytest.approx(k_mean * g["s"], rel=1e-9)
    assert not gates(r, "single_run")                         # 런별은 보고로 내려간다
    assert len([x for x in r["reports"] if x.get("kind") == "single_run"]) == 25


def test_mean_gate_catches_shift_single_run_would_miss(env):
    base, cur, t0 = env
    _retrain_all_shrb(cur, t0, frac_sd=5.0)                    # 4.27·s < 5·s < 6.51·s
    r = C.retrain_compare(base, cur, status(t0, SHRB, cur), TOL)
    assert r["exit_code"] == 1
    assert [g["metric"] for g in gates(r, "mean", "FAIL")] == ["top1"]
    per_run = [x for x in r["reports"] if x.get("kind") == "single_run" and x["metric"] == "top1"]
    assert per_run and all(x["status"] == "REPORT" for x in per_run)    # 단일 런 기준으로는 통과였을 것


def test_sign_gate_flip_fails(env):
    base, cur, t0 = env
    retrain(cur, "c100v_shrb", t0 + 200, silhouette=lambda v: v - 0.05)    # shrb42 < softmax42 (0.2460)
    aggregate(cur, t0 + 300, lambda d: bump_core(d, "shrb", 42, 0.1))
    r = C.retrain_compare(base, cur, status(t0, ["c100v_shrb"], cur), TOL)
    assert r["exit_code"] == 1
    flips = gates(r, "sign", "FAIL")
    assert [g["group"] for g in flips] == ["shrb_vs_softmax/silhouette", "shrb_vs_mhrb/silhouette"]
    assert flips[0]["new"] == "4/5" and flips[0]["ref"] == "5/5" and flips[0]["ratio"] >= 3


def test_sign_report_only_metrics_never_gate(env):
    base, cur, t0 = env
    retrain(cur, "c100v_shrb", t0 + 200, top1=lambda v: v - 0.002)   # shrb−mhrb top1 부호 뒤집힘(보고만)
    aggregate(cur, t0 + 300, lambda d: bump_core(d, "shrb", 42, 0.1))
    r = C.retrain_compare(base, cur, status(t0, ["c100v_shrb"], cur), TOL)
    assert r["exit_code"] == 0
    rep = [x for x in r["reports"] if x.get("kind") == "sign" and x["group"] == "shrb_vs_mhrb/top1"]
    assert rep and rep[0]["new"] == "4/5" and rep[0]["status"] == "REPORT"


# --------------------------------------------------------------------------- 보고: 단일 시드·파일 변화

def test_dml_single_seed_report_and_unexpected_change(env):
    base, cur, t0 = env
    retrain(cur, "c100v_shrb", t0 + 200)
    retrain(cur, "cifar100_arcface_lr3", t0 + 210, top1=lambda v: v + 0.025)

    def mut(d):
        bump_core(d, "shrb", 42, 0.1)
        d["core"]["arcface"]["map"]["by_seed"]["42"] += 0.03
    aggregate(cur, t0 + 300, mut)
    p = cur / "results" / "c100v_mhrb_s1" / "metrics.json"       # 학습하지 않은 런이 바뀜
    m = json.loads(p.read_text(encoding="utf-8"))
    m["top1"] += 0.01
    p.write_text(json.dumps(m), encoding="utf-8")
    r = C.retrain_compare(base, cur, status(t0, ["c100v_shrb", "cifar100_arcface_lr3"], cur), TOL)
    assert r["exit_code"] == 0, gates(r, status_="FAIL") or r["unevaluated"]
    dml = {x["metric"]: x for x in r["reports"] if x.get("exp") == "cifar100_arcface_lr3"
           and x.get("kind") == "single_seed"}
    assert dml["map"]["status"] == "WARN" and dml["map"]["delta"] == pytest.approx(0.03)
    assert dml["fewshot_s1"]["status"] == "REPORT"
    assert dml["top1"]["status"] == "WARN" and dml["top1"]["delta"] == pytest.approx(0.025)   # metrics.json float 보고
    ch = {x["path"]: x for x in r["reports"] if x.get("kind") == "file_change"}
    assert ch["results/c100v_mhrb_s1/metrics.json"]["category"] == "unexpected"
    assert ch["results/c100v_mhrb_s1/metrics.json"]["status"] == "WARN"
    assert ch["results/cifar100_arcface_lr3/metrics.json"]["category"] == "trained"


# --------------------------------------------------------------------------- 학습 산출물 지문(되돌리기·덮어쓰기)

def test_metrics_restored_after_training_is_precondition_failure(env):
    """학습 뒤 metrics.json을 기준본으로 되돌리면(새 mtime) Δ=0으로 합격하던 결함."""
    base, cur, t0 = env
    retrain(cur, "c100v_shrb", t0 + 200, top1=lambda v: v + 0.001)
    aggregate(cur, t0 + 300, lambda d: bump_core(d, "shrb", 42, 0.1))
    st = status(t0, ["c100v_shrb"], cur)                       # 학습 직후 지문
    assert C.retrain_compare(base, cur, st, TOL)["exit_code"] == 0
    mp = cur / "results" / "c100v_shrb" / "metrics.json"
    shutil.copyfile(base / "results" / "c100v_shrb" / "metrics.json", mp)   # 되돌림: mtime은 지금(신선해 보임)
    assert mp.stat().st_mtime > t0 + 100
    r = C.retrain_compare(base, cur, st, TOL)
    assert r["exit_code"] == 2 and r["pass"] is False and r["gates"] == []
    assert any("metrics.json이 학습 직후 기록" in x for x in r["precondition"]["reasons"])


def test_identical_to_baseline_needs_fingerprint(env):
    base, cur, t0 = env
    retrain(cur, "c100v_shrb", t0 + 200)                       # 내용이 기준본과 같다
    aggregate(cur, t0 + 300, lambda d: bump_core(d, "shrb", 42, 0.1))
    r = C.retrain_compare(base, cur, status(t0, ["c100v_shrb"]), TOL)       # 지문 없는(이전 형식) 상태
    assert r["exit_code"] == 2 and any("기준본과 내용이 같고" in x for x in r["precondition"]["reasons"])
    r = C.retrain_compare(base, cur, status(t0, ["c100v_shrb"], cur), TOL)  # 학습 직후 지문과 같으면 비트 동일 재현
    assert r["exit_code"] == 0
    assert [x["status"] for x in r["reports"] if x.get("kind") == "identical"] == ["WARN"]


def test_npz_tampering_and_not_rewritten_are_precondition_failures(env):
    base, cur, t0 = env
    retrain(cur, "c100v_shrb", t0 + 200, top1=lambda v: v + 0.001)
    aggregate(cur, t0 + 300, lambda d: bump_core(d, "shrb", 42, 0.1))
    npz = cur / "results" / "c100v_shrb" / "embeddings.npz"
    st = status(t0, ["c100v_shrb"], cur)
    fp = st["items"][0]["artifacts"]["c100v_shrb"]
    npz.write_bytes(b"replaced after training")               # 학습 뒤 바뀜
    r = C.retrain_compare(base, cur, st, TOL)
    assert r["exit_code"] == 2 and any("embeddings.npz가 학습 직후 기록" in x for x in r["precondition"]["reasons"])
    npz.unlink()                                              # 학습 직후엔 있었는데 사라짐
    r = C.retrain_compare(base, cur, st, TOL)
    assert r["exit_code"] == 2 and any("지금은 없음" in x for x in r["precondition"]["reasons"])
    npz.write_bytes(b"fake npz c100v_shrb")                   # 원래 내용·시각으로 복구(집계보다 새 npz면 집계가 낡음)
    _set_mtime(npz, t0 + 200)
    assert C.retrain_compare(base, cur, st, TOL)["exit_code"] == 0
    fp["embeddings.npz"]["rewritten"] = False                 # 학습이 npz를 다시 쓰지 않았다고 기록됨
    r = C.retrain_compare(base, cur, st, TOL)
    assert r["exit_code"] == 2 and any("학습 전 파일" in x for x in r["precondition"]["reasons"])
    fp["embeddings.npz"] = None                               # 학습 직후엔 없었는데 지금은 있음
    r = C.retrain_compare(base, cur, st, TOL)
    assert r["exit_code"] == 2 and any("학습 직후에는 없었는데" in x for x in r["precondition"]["reasons"])


def test_npz_older_than_start_without_fingerprint(env):
    base, cur, t0 = env
    retrain(cur, "c100v_shrb", t0 + 200, top1=lambda v: v + 0.001)
    aggregate(cur, t0 + 300, lambda d: bump_core(d, "shrb", 42, 0.1))
    assert C.retrain_compare(base, cur, status(t0, ["c100v_shrb"]), TOL)["exit_code"] == 0
    _set_mtime(cur / "results" / "c100v_shrb" / "embeddings.npz", t0)       # 실행 시작(t0+100)보다 이전
    r = C.retrain_compare(base, cur, status(t0, ["c100v_shrb"]), TOL)
    assert r["exit_code"] == 2 and any("embeddings.npz mtime" in x for x in r["precondition"]["reasons"])


# --------------------------------------------------------------------------- 실행 단위(합친 이력의 성공·실패)

def _run(t, items, ok=True, code=None):
    return {"started": datetime.fromtimestamp(t).isoformat(), "started_ns": int(t * 1e9), "items": items,
            "trained": [e for it in items for e in it.get("trained", [])], "pass": ok,
            "exit_code": (0 if ok else 1) if code is None else code, "_source": f"runs/x/{int(t)}.json"}


def _titem(iid, expected, trained, ok, t, cur=None):
    it = {"id": iid, "kind": "train", "required": True, "exit": 0 if ok else 1, "ok": ok, "expected": list(expected),
          "cached": [], "trained": list(trained), "started_ns": int(t * 1e9)}
    if cur is not None:
        it["artifacts"] = artifacts(cur, trained)
    return it


def _merged(runs):
    """nmfc_cli.merged_status와 같은 모양: 최상위 = 마지막 실행, items·trained 합집합, 가장 이른 시작, runs."""
    m = dict(runs[-1])
    m["items"] = [it for r in runs for it in r["items"]]
    m["trained"] = list(dict.fromkeys(e for r in runs for e in r["trained"]))
    m["started_ns"], m["started"] = runs[0]["started_ns"], runs[0]["started"]
    m["runs"] = runs
    return m


def test_failed_latest_run_blocks_pass_from_history(env):
    """마지막 train이 실패(학습 0런)해도 이전 실행의 런만으로 PASS하던 결함."""
    base, cur, t0 = env
    retrain(cur, "syn_moons_nmfc", t0 + 200, test_top1=lambda v: v + 0.001)
    good = _run(t0 + 150, [_titem("T1.1", ["syn_moons_nmfc"], ["syn_moons_nmfc"], True, t0 + 150, cur)])
    assert C.retrain_compare(base, cur, _merged([good]), TOL)["exit_code"] == 0
    bad = _run(t0 + 400, [_titem("T5:c100v_shrb", ["c100v_shrb"], [], False, t0 + 400)], ok=False)
    r = C.retrain_compare(base, cur, _merged([good, bad]), TOL)
    assert r["exit_code"] == 2 and r["pass"] is False and r["gates"] == []
    assert any("마지막 train/reproduce 실행이 성공하지 않음" in x and "c100v_shrb" in x
               for x in r["precondition"]["reasons"])
    assert [x["state"] for x in r["runs"]] == ["ok", "latest_failed"]
    assert r["runs"][1]["expected_to_train"] == ["c100v_shrb"] and r["runs"][1]["trained"] == []
    assert r["runs"][1]["failed_items"][0]["unfinished"] == ["c100v_shrb"]
    assert C.retrain_compare(base, cur, bad, TOL)["exit_code"] == 2              # --latest-only 모양
    # 중단(pass None, exit 130, 끝나지 않은 항목)도 마지막이면 막는다
    stop = _run(t0 + 400, [{**_titem("T5:c100v_shrb", ["c100v_shrb"], [], False, t0 + 400), "exit": None,
                            "ok": None}], ok=None, code=130)
    r = C.retrain_compare(base, cur, _merged([good, stop]), TOL)
    assert r["exit_code"] == 2 and r["runs"][1]["state"] == "latest_failed"


def test_earlier_failure_must_be_resolved_by_later_success(env):
    base, cur, t0 = env
    retrain(cur, "syn_moons_nmfc", t0 + 300, test_top1=lambda v: v + 0.001)
    bad = _run(t0 + 150, [_titem("T1:syn_moons_nmfc", ["syn_moons_nmfc"], [], False, t0 + 150),
                          {"id": "S3.retrieval", "kind": "step", "required": True, "exit": 1, "ok": False}], ok=False)
    good = _run(t0 + 250, [_titem("T1:syn_moons_nmfc", ["syn_moons_nmfc"], ["syn_moons_nmfc"], True, t0 + 250, cur),
                           {"id": "S3.retrieval", "kind": "step", "required": True, "exit": 0, "ok": True}])
    r = C.retrain_compare(base, cur, _merged([bad, good]), TOL)
    assert r["exit_code"] == 0, r["precondition"]
    assert [x["state"] for x in r["runs"]] == ["resolved", "ok"]
    # 실패한 런을 뒤에서 다시 해내지 않았으면(다른 런만 학습) 해소되지 않는다
    other = _run(t0 + 250, [_titem("T1:syn_moons_nmfc", ["syn_moons_nmfc"], ["syn_moons_nmfc"], True, t0 + 250, cur)])
    bad2 = _run(t0 + 150, [_titem("T5:c100v_shrb", ["c100v_shrb"], [], False, t0 + 150)], ok=False)
    r = C.retrain_compare(base, cur, _merged([bad2, other]), TOL)
    assert r["exit_code"] == 2 and [x["state"] for x in r["runs"]] == ["unresolved", "ok"]
    assert any("해소되지 않음" in x and "c100v_shrb" in x for x in r["precondition"]["reasons"])


def test_latest_success_without_training_does_not_block(env):
    """학습 뒤 집계만 다시 돌린 성공 실행(예: reproduce --from S3)이 마지막이어도 앞 실행의 학습으로 판정한다."""
    base, cur, t0 = env
    retrain(cur, "syn_moons_nmfc", t0 + 200, test_top1=lambda v: v + 0.001)
    good = _run(t0 + 150, [_titem("T1.1", ["syn_moons_nmfc"], ["syn_moons_nmfc"], True, t0 + 150, cur)])
    agg = _run(t0 + 500, [{"id": "S3.retrieval", "kind": "step", "required": True, "exit": 0, "ok": True,
                           "trained": []}])
    r = C.retrain_compare(base, cur, _merged([good, agg]), TOL)
    assert r["exit_code"] == 0 and [x["state"] for x in r["runs"]] == ["ok", "ok"]
