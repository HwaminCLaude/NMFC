"""regen 판정 테스트 — JSON 경로 규칙, regen_compare, regen_label.

모든 비교기 테스트는 '일부러 망가뜨린 입력에서 FAIL'을 함께 확인한다.
"""

import collections
import copy
import json
import shutil
import sys
from pathlib import Path

import pytest

DOCKER = Path(__file__).resolve().parents[1]
REPO = DOCKER.parent
FIX = Path(__file__).resolve().parent / "fixtures"
if str(DOCKER) not in sys.path:
    sys.path.insert(0, str(DOCKER))

import nmfc_compare as C  # noqa: E402

TOL = C.load_yaml(DOCKER / "tolerances.yaml")
REF_DEVICE = TOL["reference_device"]
GPU_REF = {"torch": "2.11.0+cu128", "cuda": "12.8", "cudnn": 91900, "gpu_name": "NVIDIA GeForce RTX 5080",
           "capability": (12, 0), "gpu_mem": 16_000_000_000, "device": "cuda"}
CPU_DEV = {"torch": "2.11.0+cpu", "cuda": None, "cudnn": None, "gpu_name": None, "capability": None,
           "gpu_mem": None, "device": "cpu"}
RE, SC, SR = "results/retrieval_eval.json", "results/stats_ci.json", "results/seed_robustness.json"
Q = 1.0 / 198000  # few-shot 1 양자 = 1/(20 에피소드 × 9900 질의)


def fx(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def ref(rel):
    return fx("ref/" + Path(rel).name)


def cmp(rel, new, mode, base=None):
    return C.compare_json_rules(rel, ref(rel) if base is None else base, new, mode, TOL["json"][rel]["rules"])


def fails(fs):
    return [f for f in fs if f["status"] == "FAIL"]


def label(findings, device):
    return C.regen_label({"findings": list(findings)}, device, REF_DEVICE)


def row(data, exp):
    return next(r for r in data if r["exp"] == exp)


# --------------------------------------------------------------------------- 규칙 파일·문법

def test_tolerances_numbers_are_floats():
    for rel, spec in TOL["json"].items():
        for r in spec["rules"]:
            for mode in ("ref", "other", "all"):
                chk = r.get(mode)
                if isinstance(chk, dict) and "abs" in chk:
                    assert isinstance(chk["abs"], float), (rel, r)
            cr = C._compile_rules([r])[0]
            for mode in ("ref", "other"):
                t = cr[mode]["tol"]
                assert t is None or isinstance(t, float), (rel, r)
    assert TOL["reference_device"]["capability"] == [12, 0]
    assert "macros" not in TOL


def test_load_yaml_coerces_scientific_strings(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("a: 1e-5\nb: 1.0e5\nc: 6e1234\nd: '2.11.0+cu128'\ne: 1.0e-5\n", encoding="utf-8")
    d = C.load_yaml(p)
    assert d["a"] == 1e-5 and isinstance(d["a"], float)
    assert d["b"] == 1e5 and isinstance(d["b"], float)
    assert d["c"] == "6e1234"           # 점·부호 없는 지수 꼴(해시 조각 등)은 건드리지 않는다
    assert d["d"] == "2.11.0+cu128" and d["e"] == 1e-5


def test_glob_semantics():
    g = C._glob_match
    assert g("*/purity", ("cifar100_deepmfa", "purity"))
    assert not g("*/purity", ("a", "b", "purity"))             # '*'는 성분 하나
    assert g("**/positive_episodes", ("fewshot_k1_paired_delta", "c100v_mhrb", "positive_episodes"))
    assert not g("*/positive_episodes", ("fewshot_k1_paired_delta", "c100v_mhrb", "positive_episodes"))
    assert g("**/n", ("n",)) and g("**/n", ("core", "shrb", "map", "n"))
    assert not g("**/n", ("seeds",)) and not g("**/n", ("core", "shrb", "map", "n_pos"))
    assert g("**", ()) and g("**", ("a", "b"))
    assert g("fewshot_k1*/**", ("fewshot_k1_paired_delta", "x", "sd"))
    assert g("*/recall@*", ("c100v_shrb", "recall@1"))
    assert not g("*/mAP", ("c100v_shrb", "map"))               # 대소문자 구분
    assert g("core/*/map/**", ("core", "shrb", "map", "by_seed", "42"))


def test_first_matching_rule_wins():
    # **/n은 core/*/map/** 보다 위 → map/n은 허용오차가 아니라 exact
    base = ref(SR)
    new = copy.deepcopy(base)
    new["core"]["shrb"]["map"]["n"] = 5.0  # 형 변화(int→float)도 FAIL
    fs = cmp(SR, new, "other")
    assert [f["path"] for f in fails(fs)] == ["core/shrb/map/n"]


# --------------------------------------------------------------------------- 실측 픽스처

@pytest.mark.parametrize("rel,fixture", [
    (RE, "retrieval_eval.cpu.json"), (RE, "retrieval_eval.cpu_run1.json"), (RE, "retrieval_eval.cpu_run2.json"),
    (SC, "stats_ci.cpu.json"), (SR, "seed_robustness.cpu.json")])
def test_cpu_fixtures_pass_tolerance_in_other_mode(rel, fixture):
    fs = cmp(rel, fx(fixture), "other")
    assert fs, "CPU 재생성본은 기준본과 달라야 한다(공허한 통과 방지)"
    assert not fails(fs), fails(fs)[:3]
    assert label(fs, CPU_DEV) == "PASS_TOLERANCE"
    assert label(fs, GPU_REF) == "PASS_TOLERANCE"  # 허용오차 안의 차이는 PASS_EXACT가 아니다


@pytest.mark.parametrize("rel,fixture", [
    (RE, "retrieval_eval.cpu.json"), (RE, "retrieval_eval.cpu_run2.json"),
    (SC, "stats_ci.cpu.json"), (SR, "seed_robustness.cpu.json")])
def test_cpu_fixtures_fail_in_ref_mode(rel, fixture):
    fs = cmp(rel, fx(fixture), "ref")
    assert fails(fs)
    assert label(fs, GPU_REF) == "FAIL"


def test_cpu_fixture_drift_magnitudes_are_as_measured():
    fs = cmp(RE, fx("retrieval_eval.cpu.json"), "other")
    rec = [f["delta"] for f in fs if f["path"].endswith("recall@1")]
    maps = [f["delta"] for f in fs if f["path"].endswith("/mAP")]
    assert max(rec) == pytest.approx(2e-4, abs=1e-9) and max(maps) < 2.44e-6
    assert {f["path"] for f in fs if f["status"] == "REPORT"} == {"cifar100_deepmfa/purity", "cifar100_deepmfa/nmi"}


def test_gpu_run1_only_purity_nmi_reported_and_pass_exact():
    fs = cmp(RE, fx("retrieval_eval.gpu_run1.json"), "ref")
    assert {f["path"] for f in fs} == {"cifar100_deepmfa/purity", "cifar100_deepmfa/nmi"}
    assert all(f["status"] == "REPORT" and f["gate"] is False for f in fs)
    assert label(fs, GPU_REF) == "PASS_EXACT"
    assert label(fs, CPU_DEV) == "PASS_TOLERANCE"   # 장치가 reference가 아니면 EXACT 불가


def test_gpu_run2_identical():
    fs = cmp(RE, fx("retrieval_eval.gpu_run2.json"), "ref")
    assert fs == []
    assert label(fs, GPU_REF) == "PASS_EXACT"


# --------------------------------------------------------------------------- retrieval_eval 변형

def test_row_deleted_fails():
    new = ref(RE)
    del new[3]
    for mode in ("ref", "other"):
        fs = cmp(RE, new, mode)
        assert [(f["path"], f["status"]) for f in fs] == [("cifar100_nmfc", "FAIL")]
        assert label(fs, GPU_REF) == "FAIL"


def test_duplicate_or_missing_row_key_fails():
    new = ref(RE)
    new.append(copy.deepcopy(new[0]))
    assert any(f["check"] == "rows_key" and f["status"] == "FAIL" for f in cmp(RE, new, "other"))
    new = ref(RE)
    del new[0]["exp"]
    assert fails(cmp(RE, new, "other"))


def test_recall_plus_1e4_tolerated_in_other_only():
    new = ref(RE)
    row(new, "cifar100_softmax_lr3")["recall@1"] += 1e-4
    other = cmp(RE, new, "other")
    assert [(f["path"], f["status"]) for f in other] == [("cifar100_softmax_lr3/recall@1", "PASS")]
    assert label(other, CPU_DEV) == "PASS_TOLERANCE"
    assert label(other, GPU_REF) == "PASS_TOLERANCE"
    assert fails(cmp(RE, new, "ref"))


def test_recall_beyond_5e4_fails_other():
    new = ref(RE)
    row(new, "cifar100_softmax_lr3")["recall@1"] += 6e-4
    assert [f["path"] for f in fails(cmp(RE, new, "other"))] == ["cifar100_softmax_lr3/recall@1"]


def test_map_plus_1e3_fails_other_but_5e6_passes():
    new = ref(RE)
    row(new, "c100v_shrb")["mAP"] += 1e-3
    assert [f["path"] for f in fails(cmp(RE, new, "other"))] == ["c100v_shrb/mAP"]
    new = ref(RE)
    row(new, "c100v_shrb")["mAP"] += 5e-6
    assert not fails(cmp(RE, new, "other"))


def test_top1_any_change_fails_even_other():
    new = ref(RE)
    row(new, "c100v_shrb")["top1"] += 1e-9
    assert fails(cmp(RE, new, "other"))


def test_purity_large_change_is_report_only():
    new = ref(RE)
    row(new, "c100v_shrb")["purity"] += 0.05
    for mode in ("ref", "other"):
        fs = cmp(RE, new, mode)
        assert len(fs) == 1 and fs[0]["status"] == "REPORT" and fs[0]["gate"] is False
    assert cmp(RE, new, "other")[0]["data"]["within_tol"] is False
    assert label(cmp(RE, new, "ref"), GPU_REF) == "PASS_EXACT"


def test_type_and_structure_changes_fail():
    for mutate in (lambda d: row(d, "c100v_shrb").__setitem__("mAP", "0.752"),     # float → str
                   lambda d: row(d, "c100v_shrb").__setitem__("purity", None),      # 보고 규칙도 형 변화는 FAIL
                   lambda d: row(d, "c100v_shrb").__setitem__("group", "X"),
                   lambda d: row(d, "c100v_shrb").__setitem__("extra_key", 1.0),
                   lambda d: row(d, "c100v_shrb").pop("nmi")):
        new = ref(RE)
        mutate(new)
        assert fails(cmp(RE, new, "other")), mutate


def test_row_order_change_is_warn_and_blocks_exact():
    new = list(reversed(ref(RE)))
    fs = cmp(RE, new, "ref")
    assert [f["status"] for f in fs] == ["WARN"] and fs[0]["check"] == "order"
    assert label(fs, GPU_REF) == "PASS_TOLERANCE"


# --------------------------------------------------------------------------- stats_ci 변형

def test_stats_ci_verification_warn_then_fail():
    new = ref(SC)
    new["verification_auc"]["c100v_shrb"]["auc"] += 5e-7
    fs = cmp(SC, new, "other")
    assert [f["status"] for f in fs] == ["WARN"]
    assert fails(cmp(SC, new, "ref"))
    new["verification_auc"]["c100v_shrb"]["auc"] += 2e-6
    assert fails(cmp(SC, new, "other"))


def test_stats_ci_fewshot_one_quantum_warn_three_fail():
    new = ref(SC)
    new["fewshot_k1"]["c100v_shrb"]["mean"] += Q
    assert [f["status"] for f in cmp(SC, new, "other")] == ["WARN"]
    new["fewshot_k1"]["c100v_shrb"]["mean"] += 2 * Q
    assert fails(cmp(SC, new, "other"))


@pytest.mark.parametrize("path,value", [
    (("map_delta_vs_shrb", "c100v_mhrb", "significant"), False),
    (("fewshot_k1_paired_delta", "c100v_mhrb", "positive_episodes"), 17),
    (("fewshot_k1_paired_delta", "c100v_mhrb", "n_episodes"), 19),
    (("verification_delta_vs_shrb", "c100v_mhrb", "exceeds_band"), True),
    (("n_boot",), 999)])
def test_stats_ci_exact_items_fail(path, value):
    new = ref(SC)
    d = new
    for p in path[:-1]:
        d = d[p]
    d[path[-1]] = value
    assert fails(cmp(SC, new, "other"))


def test_stats_ci_map_ci_tolerance():
    new = ref(SC)
    new["map_delta_vs_shrb"]["cifar100_softmax_lr3"]["ci95"][0] += 9e-6
    assert not fails(cmp(SC, new, "other"))
    new["map_delta_vs_shrb"]["cifar100_softmax_lr3"]["ci95"][0] += 2e-6
    assert fails(cmp(SC, new, "other"))


# --------------------------------------------------------------------------- seed_robustness 변형

def test_seed_robustness_mutations():
    base = ref(SR)
    cases = [
        (lambda d: d["core"]["shrb"]["map"].__setitem__("n", 4), True),
        (lambda d: d["deltas"]["shrb_vs_softmax"]["map"].__setitem__("n_pos", 4), True),
        (lambda d: d["core"]["shrb"]["map"]["by_seed"].__setitem__("0", d["core"]["shrb"]["map"]["by_seed"]["0"] + 2e-5), True),
        (lambda d: d["core"]["shrb"]["map"]["by_seed"].__setitem__("0", d["core"]["shrb"]["map"]["by_seed"]["0"] + 5e-6), False),
        (lambda d: d["core"]["shrb"]["fewshot_s1"]["by_seed"].__setitem__("0", d["core"]["shrb"]["fewshot_s1"]["by_seed"]["0"] + Q), False),
        (lambda d: d["core"]["shrb"]["top1"].__setitem__("mean", d["core"]["shrb"]["top1"]["mean"] + 1e-9), True),
        (lambda d: d["band"].__setitem__("paper_band", 0.02), True),
        (lambda d: d["seeds"].append(4), True),
    ]
    for mutate, should_fail in cases:
        new = copy.deepcopy(base)
        mutate(new)
        fs = cmp(SR, new, "other")
        assert fs, mutate
        assert bool(fails(fs)) is should_fail, (should_fail, fs)


# --------------------------------------------------------------------------- regen_compare 전체

def _make_tree(root: Path, retrieval, stats, seed, crlf=False):
    (root / "results" / "openset_noise_grid").mkdir(parents=True)
    for rel, data in ((RE, retrieval), (SC, stats), (SR, seed)):
        (root / rel).write_bytes(json.dumps(data, indent=1, ensure_ascii=False).encode("utf-8"))
    shutil.copyfile(FIX / "ref" / "fewshot_verify_eval.json", root / "results" / "fewshot_verify_eval.json")
    grid = REPO / "results" / "openset_noise_grid"
    for p in grid.iterdir():
        if p.suffix in (".json", ".png"):
            shutil.copyfile(p, root / "results" / "openset_noise_grid" / p.name)
    csv_bytes = (grid / "master.csv").read_bytes()
    if crlf:
        csv_bytes = csv_bytes.replace(b"\n", b"\r\n")
    (root / "results" / "openset_noise_grid" / "master.csv").write_bytes(csv_bytes)


@pytest.fixture
def regen_trees(tmp_path):
    base, cur = tmp_path / "base", tmp_path / "cur"
    _make_tree(base, ref(RE), ref(SC), ref(SR))
    _make_tree(cur, fx("retrieval_eval.cpu.json"), fx("stats_ci.cpu.json"), fx("seed_robustness.cpu.json"),
               crlf=True)
    return base, cur


def test_regen_compare_cpu_end_to_end(regen_trees):
    base, cur = regen_trees
    r = C.regen_compare(base, cur, "other", TOL)
    assert not [f for f in r["findings"] if f["status"] == "FAIL"]
    assert r["counts"]["json"]["files"] == 5                       # 선택(ckpt) 파일은 N/A
    assert {f["file"] for f in r["findings"] if f["status"] == "N/A"} == {
        "results/near_ood.json", "results/calibration_eval.json", "results/c100v_softmax_open_bank/metrics.json",
        "results/c100v_shrb_open_bank/metrics.json", "results/openset_detection/config_used.json",
        "results/openset_detection/auroc_table.md"}
    assert all(f["gate"] is False for f in r["findings"] if f["status"] == "N/A")
    assert r["counts"]["files"]["cells"] > 0 and r["counts"]["files"]["identical_cells"] == r["counts"]["files"]["cells"]
    assert C.regen_label(r, CPU_DEV, REF_DEVICE) == "PASS_TOLERANCE"
    rr = C.regen_compare(base, cur, "ref", TOL)
    assert C.regen_label(rr, GPU_REF, REF_DEVICE) == "FAIL"


def test_regen_compare_identical_is_pass_exact_only_on_reference(tmp_path):
    base, cur = tmp_path / "base", tmp_path / "cur"
    _make_tree(base, ref(RE), ref(SC), ref(SR))
    _make_tree(cur, fx("retrieval_eval.gpu_run1.json"), ref(SC), ref(SR), crlf=True)
    r = C.regen_compare(base, cur, C.regen_mode(GPU_REF, REF_DEVICE), TOL)
    assert C.regen_mode(GPU_REF, REF_DEVICE) == "ref" and C.regen_mode(CPU_DEV, REF_DEVICE) == "other"
    assert C.regen_label(r, GPU_REF, REF_DEVICE) == "PASS_EXACT"
    assert C.regen_label(r, {**GPU_REF, "cudnn": 91000}, REF_DEVICE) == "PASS_TOLERANCE"
    assert C.regen_label(r, {**GPU_REF, "capability": [8, 9]}, REF_DEVICE) == "PASS_TOLERANCE"
    assert C.regen_label(r, {**GPU_REF, "device": "cpu"}, REF_DEVICE) == "PASS_TOLERANCE"


def test_regen_compare_broken_inputs_fail(regen_trees):
    base, cur = regen_trees
    p = cur / "results" / "openset_noise_grid" / "master.csv"
    p.write_bytes(p.read_bytes().replace(b"0.8301205625", b"0.8301205626", 1))
    r = C.regen_compare(base, cur, "other", TOL)
    assert any(f["check"] == "csv_cells" and f["status"] == "FAIL" for f in r["findings"])
    (cur / "results" / "openset_noise_grid" / "master.json").unlink()
    r = C.regen_compare(base, cur, "other", TOL)
    assert any(f["file"].endswith("master.json") and f["status"] == "FAIL" for f in r["findings"])
    (cur / "results" / "openset_noise_grid" / "did_curve.png").unlink()
    r = C.regen_compare(base, cur, "other", TOL)
    assert any(f["file"].endswith("did_curve.png") and f["status"] == "FAIL" for f in r["findings"])


def test_regen_compare_missing_baseline_is_fail_not_vacuous(tmp_path):
    base, cur = tmp_path / "base", tmp_path / "cur"
    _make_tree(cur, ref(RE), ref(SC), ref(SR))
    base.mkdir()
    r = C.regen_compare(base, cur, "ref", TOL)
    assert C.regen_label(r, GPU_REF, REF_DEVICE) == "FAIL"


def test_noise_grid_png_difference_is_report(regen_trees):
    from PIL import Image

    base, cur = regen_trees
    p = cur / "results" / "openset_noise_grid" / "gap_curve.png"
    with Image.open(p) as im:
        im = im.convert("RGBA")
        im.putpixel((0, 0), (1, 2, 3, 255))
        im.save(p)
    r = C.regen_compare(base, cur, "other", TOL)
    rep = [f for f in r["findings"] if f["file"].endswith("gap_curve.png")]
    assert len(rep) == 1 and rep[0]["status"] == "REPORT" and rep[0]["gate"] is False
    assert C.regen_label(r, CPU_DEV, REF_DEVICE) == "PASS_TOLERANCE"


# --------------------------------------------------------------------------- ckpt 평가기 산출물: 보고하되 PASS_EXACT 차단

CKPT_JSON = ("results/near_ood.json", "results/calibration_eval.json", "results/c100v_softmax_open_bank/metrics.json",
             "results/c100v_shrb_open_bank/metrics.json", "results/openset_detection/config_used.json")
CKPT_TABLE = "results/openset_detection/auroc_table.md"


def test_blocks_exact_report_rule():
    chk = C._parse_check({"report": True, "blocks_exact": True})
    assert chk["kind"] == "report" and chk["blocks_exact"] is True
    assert C._parse_check("report").get("blocks_exact", False) is False
    rules = [{"path": "**", "all": {"report": True, "blocks_exact": True}}]
    fs = C.compare_json_rules("x.json", {"a": 0.5, "s": "k"}, {"a": 0.55, "s": "k"}, "ref", rules)
    assert [(f["status"], f["gate"]) for f in fs] == [("REPORT", True)]
    assert label(fs, GPU_REF) == "PASS_TOLERANCE"                           # FAIL은 아니지만 PASS_EXACT도 아님
    plain = C.compare_json_rules("x.json", {"a": 0.5}, {"a": 0.55}, "ref", [{"path": "**", "all": "report"}])
    assert label(plain, GPU_REF) == "PASS_EXACT"                            # 순수 보고(purity·nmi형)는 그대로
    assert fails(C.compare_json_rules("x.json", {"s": "k"}, {"s": "z"}, "ref", rules))   # 문자열은 정확 일치


@pytest.fixture
def ckpt_trees(tmp_path):
    """저장소의 ckpt 평가기 산출물(커밋본)을 기준본·현재 양쪽에 둔 트리."""
    base, cur = tmp_path / "base", tmp_path / "cur"
    for root in (base, cur):
        _make_tree(root, ref(RE), ref(SC), ref(SR))
        for rel in CKPT_JSON + (CKPT_TABLE,):
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / rel, root / rel)
    return base, cur


def _edit_json(path: Path, fn) -> None:
    d = json.loads(path.read_text(encoding="utf-8"))
    fn(d)
    path.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")


@pytest.mark.parametrize("rel,fn", [
    ("results/near_ood.json",
     lambda d: d["ood10v_shrb_s1"].__setitem__("far_auc", d["ood10v_shrb_s1"]["far_auc"] + 0.05)),
    ("results/calibration_eval.json", lambda d: d["rows"][0].__setitem__("ece", d["rows"][0]["ece"] + 0.05)),
    ("results/c100v_softmax_open_bank/metrics.json",
     lambda d: d.__setitem__("open_top1_all", d["open_top1_all"] + 0.05)),
])
def test_ckpt_output_difference_blocks_pass_exact(ckpt_trees, rel, fn):
    """ckpt 평가기 산출물만 달라도 PASS_EXACT가 나던 결함 → 보고(FAIL 아님) + PASS_TOLERANCE."""
    base, cur = ckpt_trees
    assert C.regen_label(C.regen_compare(base, cur, "ref", TOL), GPU_REF, REF_DEVICE) == "PASS_EXACT"
    _edit_json(cur / rel, fn)
    r = C.regen_compare(base, cur, "ref", TOL)
    hit = [f for f in r["findings"] if f["file"] == rel]
    assert hit and all(f["status"] == "REPORT" and f["gate"] is True for f in hit), hit
    assert C.regen_label(r, GPU_REF, REF_DEVICE) == "PASS_TOLERANCE"


def test_calibration_rows_pair_by_exp_and_variant(ckpt_trees):
    """calibration_eval.json 행은 exp만으로는 겹친다(raw·ts 두 행) → (exp, variant)로 짝짓는다.
    가운데 행 하나를 지워도 뒤 행들이 한 칸씩 밀려 이웃 행과 비교되지 않는다."""
    base, cur = ckpt_trees
    rel = "results/calibration_eval.json"
    rows = json.loads((cur / rel).read_text(encoding="utf-8"))["rows"]
    keys = [(r["exp"], r["variant"]) for r in rows]
    assert len(set(keys)) == len(keys) > len({r["exp"] for r in rows})          # exp만으로는 짝지을 수 없다
    mid = len(rows) // 2
    gone, last = keys[mid], keys[-1]

    def mutate(d):
        del d["rows"][mid]                                                    # 가운데 행 삭제
        d["rows"][-1]["ece"] += 0.05                                          # 마지막 행 값 하나만 바꿈
    _edit_json(cur / rel, mutate)
    hit = [f for f in C.regen_compare(base, cur, "ref", TOL)["findings"] if f["file"] == rel]
    miss = [f for f in hit if f["check"] == "structure"]
    leaf = [f for f in hit if f["check"] != "structure"]
    assert len(hit) == 2, hit                                                 # 인덱스로 짝지으면 밀린 행마다 차이가 난다
    assert len(miss) == 1 and all(k in miss[0]["path"] for k in gone), miss
    assert len(leaf) == 1 and all(k in leaf[0]["path"] for k in last) and leaf[0]["path"].endswith("ece"), leaf
    assert leaf[0]["status"] == "REPORT" and leaf[0]["gate"] is True


def test_openset_detection_table_difference_blocks_pass_exact(ckpt_trees):
    base, cur = ckpt_trees
    p = cur / CKPT_TABLE
    assert b"0.8301" in p.read_bytes()
    p.write_bytes(p.read_bytes().replace(b"0.8301", b"0.8302", 1))
    r = C.regen_compare(base, cur, "ref", TOL)
    assert [(f["status"], f["gate"]) for f in r["findings"] if f["file"] == CKPT_TABLE] == [("REPORT", True)]
    assert C.regen_label(r, GPU_REF, REF_DEVICE) == "PASS_TOLERANCE"
    p.unlink()                                                              # 선택 파일이 없으면 N/A(게이트 아님)
    r = C.regen_compare(base, cur, "ref", TOL)
    assert [(f["status"], f["gate"]) for f in r["findings"] if f["file"] == CKPT_TABLE] == [("N/A", False)]
