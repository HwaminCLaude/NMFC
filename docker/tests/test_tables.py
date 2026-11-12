"""결과표(nmfc_tables)와 nmfc_cli의 결과표 훅.

쓰기는 모두 tmp_path에만 한다(저장소의 results/·fixtures는 읽기만).
- 작성기: CSV BOM·열, Markdown 제목 줄·건수 줄, 판정 값 어휘(PASS·FAIL·REPORT·N/A), 빈 입력, 잘못된 판정·열 거부
- 상태 → 판정: PASS·FAIL은 그대로, REPORT·WARN·INFO → REPORT, N/A·건너뜀·없음 → N/A(판정하지 않은 행은 PASS가 아니다)
- 명령별 표: verify(theory_audit 픽스처), regen(같은 값 행 포함, 건너뛴 단계·다시 쓰이지 않은 파일은 파일마다 한 행),
  train(판정 없음), compare(게이트·보고·미평가·전제조건)
- 훅: import는 표준 라이브러리만, verify·train·compare 실행 뒤 <out>/<명령>/ 표, 표 실패는 종료코드 불변(WARN 검사)
"""
from __future__ import annotations

import argparse
import collections
import csv
import inspect
import io
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

DOCKER = Path(__file__).resolve().parents[1]
REPO = DOCKER.parent
FIX = Path(__file__).resolve().parent / "fixtures"
if str(DOCKER) not in sys.path:
    sys.path.insert(0, str(DOCKER))

import nmfc_cli as CLI  # noqa: E402
import nmfc_common as common  # noqa: E402
import nmfc_compare as C  # noqa: E402
import nmfc_pipeline as P  # noqa: E402
import nmfc_tables as T  # noqa: E402

BOM = b"\xef\xbb\xbf"
V = T.VERDICT
CKPT_FILES = {"results/near_ood.json", "results/calibration_eval.json", "results/c100v_softmax_open_bank/metrics.json",
              "results/c100v_shrb_open_bank/metrics.json", "results/openset_detection/config_used.json",
              "results/openset_detection/auroc_table.md"}


@pytest.fixture(autouse=True)
def _host(monkeypatch):
    monkeypatch.delenv("NMFC_IN_CONTAINER", raising=False)
    monkeypatch.delenv("NMFC_BASELINE", raising=False)
    monkeypatch.delenv("NMFC_MANIFEST_SHA256", raising=False)


def read_csv(p: Path) -> list[dict]:
    raw = Path(p).read_bytes()
    assert raw.startswith(BOM) and raw.count(BOM) == 1
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline="")))


def read_md(p: Path) -> list[str]:
    text = Path(p).read_text(encoding="utf-8")
    assert "\r" not in text
    return text.splitlines()


def tally(rows) -> dict:
    c = collections.Counter(r[V] for r in rows)
    return {v: c.get(v, 0) for v in T.VERDICTS}


# ---------------------------------------------------------------- 작성기

def test_write_table_csv_bom_and_markdown_count_line(tmp_path):
    cols = ("항목", T.COMMITTED, T.NEW, V, T.NOTE)
    rows = [{"항목": "a", T.COMMITTED: 1, T.NEW: 1.5, V: "PASS"},
            {"항목": "b|c", T.COMMITTED: "x\ny", T.NEW: None, V: "FAIL", T.NOTE: "checks/*/name"},
            {"항목": "c", V: "REPORT"}, {"항목": "d", V: "N/A"}, {"항목": "e", V: "PASS"}]
    t = T.write_table(tmp_path / "out" / "verify", "demo", "데모 표", cols, rows, notes=["메모 하나"])
    assert (t.csv, t.md) == (tmp_path / "out" / "verify" / "demo.csv", tmp_path / "out" / "verify" / "demo.md")
    assert t.count_line == "PASS 2 · FAIL 1 · REPORT 1 · N/A 1"
    assert t.counts == {"PASS": 2, "FAIL": 1, "REPORT": 1, "N/A": 1}
    assert b"\r\n" in t.csv.read_bytes()
    got = read_csv(t.csv)
    assert list(got[0]) == list(cols) and [r["항목"] for r in got] == ["a", "b|c", "c", "d", "e"]
    assert (got[0][T.COMMITTED], got[0][T.NEW], got[1][T.COMMITTED], got[1][T.NEW]) == ("1", "1.5", "x\ny", "")
    md = read_md(t.md)
    assert md[0] == "# 데모 표" and md[1] == t.count_line and md[2] == "" and md[3] == T.LEGEND
    head = md.index("| " + " | ".join(cols) + " |")             # CSV와 같은 열
    assert md[head + 1] == "|" + "---|" * len(cols) and len(md[head + 2:]) == len(rows)
    assert "| b\\|c | x<br>y |  | FAIL | checks/\\*/name |" in md   # '|'·줄바꿈·'*' 이스케이프
    assert "- 메모 하나" in md
    assert {r[V] for r in got} <= set(T.VERDICTS)


@pytest.mark.parametrize("bad", ["WARN", "pass", "PASS_EXACT", "", None])
def test_write_table_rejects_verdict_outside_vocabulary(tmp_path, bad):
    with pytest.raises(ValueError):
        T.write_table(tmp_path, "t", "t", ("a", V), [{"a": 1, V: bad}])
    assert not (tmp_path / "t.csv").exists()


def test_write_table_requires_exactly_one_verdict_column(tmp_path):
    for cols in (("a", "b"), ("a", V, V)):
        with pytest.raises(ValueError):
            T.write_table(tmp_path, "t", "t", cols, [])
    with pytest.raises(ValueError):                                 # 열에 없는 키
        T.write_table(tmp_path, "t", "t", ("a", V), [{"a": 1, "b": 2, V: "PASS"}])
    with pytest.raises(ValueError):                                 # 표 이름은 파일 이름 한 성분
        T.write_table(tmp_path, "../t", "t", ("a", V), [])


def test_write_table_empty_rows(tmp_path):
    t = T.write_table(tmp_path, "empty", "빈 표", T.GATE_COLUMNS, [])
    assert t.csv.read_bytes() == BOM + (",".join(T.GATE_COLUMNS) + "\r\n").encode("utf-8")
    md = read_md(t.md)
    assert md[1] == "PASS 0 · FAIL 0 · REPORT 0 · N/A 0" and md[-1] == "(행 없음)"


@pytest.mark.parametrize("status,want", [
    ("PASS", "PASS"), ("pass", "PASS"), ("FAIL", "FAIL"), (" fail ", "FAIL"),
    ("REPORT", "REPORT"), ("WARN", "REPORT"), ("INFO", "REPORT"), ("PASS_EXACT", "REPORT"), ("모름", "REPORT"),
    ("N/A", "N/A"), ("SKIPPED", "N/A"), ("skip", "N/A"), ("not evaluated", "N/A"), ("", "N/A"), (None, "N/A")])
def test_status_to_verdict(status, want):
    assert T.to_verdict(status) == want


def test_count_line_and_value_formatting():
    assert T.count_line({"FAIL": 2}) == "PASS 0 · FAIL 2 · REPORT 0 · N/A 0"
    assert T.count_line({"PASS": 3102, "REPORT": 4, "N/A": 5}) == "PASS 3102 · FAIL 0 · REPORT 4 · N/A 5"
    assert [T.fmt_value(v) for v in (None, True, 3, 0.1, float("nan"), "s", {"a": 1}, [])] == \
        ["null", "true", "3", "0.1", "NaN", "s", '{"a": 1}', "[]"]
    assert T.fmt_diff(T.signed_diff(0.8610720038414001, 0.8600720038414001)) == "-0.001"   # 표기끼리의 십진 차이
    assert T.signed_diff(200, 198) == -2 and T.fmt_diff(T.signed_diff(1.0, 1.0)) == "0"
    assert T.signed_diff(True, 1) is None and T.signed_diff("1", 2) is None and T.signed_diff(1.0, float("inf")) is None
    assert T.text_number_diff("0.862125", "0.862126") == "0.000001"
    assert T.text_number_diff("2.842e-14", "2.368e-14") == "-4.74e-15" and T.text_number_diff("40000", "41000") == "1000"
    assert T.text_number_diff("exp", "1") == "" and T.text_number_diff("nan", "1") == ""
    assert T.tol_label("abs 1e-06, WARN") == "abs 1e-06 (넘으면 FAIL, 안이지만 값이 다르면 REPORT)"
    assert T.tol_label("exact") == "exact" and T.tol_label(None) == ""


def test_float_platform_rule_matches_compare():
    sig = inspect.signature(C.compare_float_platform).parameters
    assert (sig["rel"].default, sig["abs_"].default) == (T.FP_REL, T.FP_ABS)
    assert tuple(C.FP_EXACT_PATHS) == T.FP_EXACT_PATHS


def test_import_uses_only_standard_library():
    """nmfc_cli --help·manifest 경로처럼 결과표 모듈도 torch·numpy·matplotlib·yaml 없이 import된다."""
    probe = ("import sys; sys.path.insert(0, sys.argv[1]); before = set(sys.modules); import nmfc_tables; "
             "new = {m.split('.')[0] for m in set(sys.modules) - before} - {'nmfc_tables'}; "
             "print('NONSTD', sorted(m for m in new if m not in sys.stdlib_module_names)); "
             "print('HEAVY', sorted(m for m in ('torch', 'numpy', 'matplotlib', 'yaml') if m in sys.modules))")
    r = subprocess.run([sys.executable, "-B", "-c", probe, str(DOCKER)], capture_output=True, encoding="utf-8",
                       errors="replace", env=common.child_env(), timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "NONSTD []" in r.stdout and "HEAVY []" in r.stdout, r.stdout


# ---------------------------------------------------------------- verify → theory_claims

def _theory(tmp_path: Path, new_obj=None, *, rewrite: bool = True):
    """기준본 = fixtures/ref/theory_audit.json. 새 실행 = t3 픽스처(또는 new_obj)를 다시 쓴 것. compare_written 판정까지."""
    base, cur = tmp_path / "base", tmp_path / "cur"
    for root in (base, cur):
        (root / "results").mkdir(parents=True)
    shutil.copyfile(FIX / "ref" / "theory_audit.json", base / "results" / "theory_audit.json")
    p = cur / "results" / "theory_audit.json"
    shutil.copyfile(FIX / "ref" / "theory_audit.json", p)
    before = {"results/theory_audit.json": {"exists": True, "mtime_ns": p.stat().st_mtime_ns, "sha256": None}}
    if rewrite:
        p.write_bytes((FIX / "theory_audit_t3.json").read_bytes() if new_obj is None
                      else json.dumps(new_obj, ensure_ascii=False).encode("utf-8"))
        t = before["results/theory_audit.json"]["mtime_ns"] + 2_000_000_000
        os.utime(p, ns=(t, t))
    written = C.compare_written(base, cur, [{"path": "results/theory_audit.json", "class": "float_platform"}], before)
    return base, cur, written


def _t3():
    return json.loads((FIX / "theory_audit_t3.json").read_text(encoding="utf-8"))


def _expected_theory_rows() -> int:
    ref = json.loads((FIX / "ref" / "theory_audit.json").read_text(encoding="utf-8"))
    return len(ref["checks"]) + 1 + len([k for k in ref if k not in ("checks", "n_passed", "n_total")])


def test_verify_table_t3_all_pass(tmp_path):
    base, cur, written = _theory(tmp_path)
    assert written[0]["status"] == "PASS"
    (t,) = T.verify_tables(tmp_path / "out" / "verify", base, cur, written)
    assert t.csv == tmp_path / "out" / "verify" / "theory_claims.csv"
    rows = read_csv(t.csv)
    assert len(rows) == _expected_theory_rows() and [r["구분"] for r in rows[:16]] == ["검사"] * 16
    assert tally(rows) == {"PASS": len(rows), "FAIL": 0, "REPORT": 0, "N/A": 0}
    assert read_md(t.md)[1] == t.count_line == f"PASS {len(rows)} · FAIL 0 · REPORT 0 · N/A 0"
    by = {r["항목"].split(":")[0]: r for r in rows}
    assert by["8"][T.DIFF] == "-4.74e-15" and "2.842e-14" in by["8"][T.COMMITTED] and "2.368e-14" in by["8"][T.NEW]
    assert by["0"][T.DIFF] == "0" and by["0"][T.NOTE] == "값 동일" and by["0"][T.COMMITTED].startswith("[passed=true]")
    assert by["invariance_maxerr"][T.DIFF].startswith("-4.7462034302") and by["R"][T.DIFF] == "0"
    total = next(r for r in rows if r["구분"] == "합계")
    assert (total[T.COMMITTED], total[T.NEW], total[T.DIFF]) == ("16/16", "16/16", "0")


def test_verify_table_value_outside_checks_fails(tmp_path):
    """check detail에 안 나오는 리프(R/R)만 바뀌어도 표에 FAIL 행이 있어야 한다."""
    bad = _t3()
    bad["R"]["R"] += 1e-6
    base, cur, written = _theory(tmp_path, bad)
    assert written[0]["status"] == "FAIL"
    (t,) = T.verify_tables(tmp_path / "out", base, cur, written)
    rows = read_csv(t.csv)
    assert [r["항목"] for r in rows if r[V] == "FAIL"] == ["R"]
    assert {r[V] for r in rows if r["구분"] == "검사"} == {"PASS"}
    assert "R/R" in next(r for r in rows if r["항목"] == "R")[T.NOTE]


def test_verify_table_flipped_check_fails(tmp_path):
    bad = _t3()
    bad["checks"][3]["passed"] = False
    base, cur, written = _theory(tmp_path, bad)
    rows = read_csv(T.verify_tables(tmp_path / "out", base, cur, written)[0].csv)
    fails = [r for r in rows if r[V] == "FAIL"]
    assert [r["항목"].split(":")[0] for r in fails] == ["3"] and "[passed=false]" in fails[0][T.NEW]


def test_verify_table_file_level_failures_mark_every_row(tmp_path):
    base, cur, written = _theory(tmp_path, rewrite=False)                 # 다시 쓰이지 않음
    rows = read_csv(T.verify_tables(tmp_path / "o1", base, cur, written)[0].csv)
    assert len(rows) == _expected_theory_rows() and {r[V] for r in rows} == {"FAIL"}
    assert all("mtime_ns 불변" in r[T.NOTE] for r in rows)
    base, cur, _ = _theory(tmp_path / "b")
    (cur / "results" / "theory_audit.json").unlink()                     # 실행 뒤 파일 없음
    written = C.compare_written(base, cur, [{"path": "results/theory_audit.json", "class": "float_platform"}],
                                {"results/theory_audit.json": {"exists": True, "mtime_ns": 1, "sha256": None}})
    rows = read_csv(T.verify_tables(tmp_path / "o2", base, cur, written)[0].csv)
    assert {r[V] for r in rows} == {"FAIL"} and {r[T.NEW] for r in rows} == {""}


def test_verify_table_without_judgement_is_report_or_na(tmp_path):
    base, cur, _ = _theory(tmp_path)
    rows = read_csv(T.verify_tables(tmp_path / "o1", base, cur, [])[0].csv)       # verify가 판정하지 않은 파일
    assert {r[V] for r in rows} == {"REPORT"} and len(rows) == _expected_theory_rows()
    rows = read_csv(T.verify_tables(tmp_path / "o2", tmp_path / "없음", tmp_path / "없음2", [])[0].csv)
    assert [(r["항목"], r[V]) for r in rows] == [(T.FILE_ROW, "N/A")]


def test_verify_hook_writes_theory_claims_without_extra_check(tmp_path, capsys):
    """verify 실행(audit 대신 t3 픽스처를 쓰는 단계) → <out>/verify/theory_claims, 검사 수는 그대로(3개)."""
    pytest.importorskip("matplotlib")
    base, wd = tmp_path / "base", tmp_path / "wd"
    for root in (base, wd):
        (root / "results").mkdir(parents=True)
        shutil.copyfile(FIX / "ref" / "theory_audit.json", root / "results" / "theory_audit.json")
    code = ("import os, shutil, sys; p = 'results/theory_audit.json'; m = os.stat(p).st_mtime_ns; "
            "shutil.copyfile(sys.argv[1], p); os.utime(p, ns=(m + 2000000000, m + 2000000000))")
    real = C.load_yaml(DOCKER / "expected.yaml")["verify"]
    exp = {"verify": {"font_assert": real["font_assert"],
                      "steps": [{"name": "audit_stub", "cmd": ["-c", code, str(FIX / "theory_audit_t3.json")],
                                 "writes": [{"path": "results/theory_audit.json", "class": "float_platform"}]}]}}
    ey = tmp_path / "expected.yaml"
    ey.write_text(json.dumps(exp, ensure_ascii=False), encoding="utf-8")          # JSON은 YAML이다
    out = tmp_path / "out"
    rc = CLI.main(["verify", "--expected", str(ey), "--workdir", str(wd), "--baseline", str(base), "--out", str(out)])
    d = json.loads((out / "verify_report.json").read_text(encoding="utf-8"))
    assert [c["name"] for c in d["checks"]] == ["한글 폰트 단언", "audit_stub: results/theory_audit.json", "예상 밖 변경 0건"]
    assert rc == (0 if all(c["status"] != "FAIL" for c in d["checks"]) else 1)
    rows = read_csv(out / "verify" / "theory_claims.csv")
    line = f"PASS {len(rows)} · FAIL 0 · REPORT 0 · N/A 0"
    assert len(rows) == _expected_theory_rows() and read_md(out / "verify" / "theory_claims.md")[1] == line
    assert line in capsys.readouterr().out                                        # 건수 줄을 출력한다
    assert [t["name"] for t in d["env"]["result_tables"]["tables"]] == ["theory_claims"]


# ---------------------------------------------------------------- regen → aggregates·table_files

TOL_SMALL = {
    "json": {
        "results/a.json": {"rows_key": "exp", "rules": [
            {"path": "*/v", "ref": "exact", "other": {"abs": 1.0e-3, "on_diff": "warn"}},
            {"path": "*/p", "all": "report"},
            {"path": "**", "all": "exact"}]},
        "results/opt/metrics.json": {"optional": True, "rules": [{"path": "**", "all": {"report": True, "blocks_exact": True}}]},
        "results/det/config_used.json": {"optional": True,
                                         "rules": [{"path": "**", "all": {"report": True, "blocks_exact": True}}]},
    },
    "files": [{"path": "results/grid/master.csv", "class": "csv_cells"},
              {"path": "results/det/table.md", "class": "exact", "optional": True, "on_diff": "report"}],
}
ROWS_A = [{"exp": "x", "v": 0.5, "p": 0.25, "s": "t"}, {"exp": "y", "v": 1.0, "p": 0.75, "s": "u"}]
MASTER = "method,exp,top1\nshrb,x,0.862125\nsoftmax,y,0.5\n"


def _regen_trees(tmp_path: Path):
    base, cur = tmp_path / "base", tmp_path / "cur"
    for root in (base, cur):
        for sub in ("opt", "det", "grid"):
            (root / "results" / sub).mkdir(parents=True)
        (root / "results" / "a.json").write_text(json.dumps(ROWS_A), encoding="utf-8")
        (root / "results" / "opt" / "metrics.json").write_text(json.dumps({"auc": 0.9, "n": 3}), encoding="utf-8")
        (root / "results" / "det" / "config_used.json").write_text(json.dumps({"seed": 1}), encoding="utf-8")
        (root / "results" / "det" / "table.md").write_text("| a |\n", encoding="utf-8")
        (root / "results" / "grid" / "master.csv").write_bytes(("﻿" + MASTER).encode("utf-8"))
    return base, cur


def test_regen_rows_include_identical_without_changing_findings(tmp_path):
    """include_identical은 findings·counts·라벨을 바꾸지 않고, 같은 리프·셀까지 행으로 준다."""
    base, cur = _regen_trees(tmp_path)
    rows_a = json.loads(json.dumps(ROWS_A))
    rows_a[1]["v"] += 5e-4                                     # 허용오차 안(on_diff warn)
    rows_a[0]["s"] = "T"                                       # 문자열 불일치 → FAIL
    (cur / "results" / "a.json").write_text(json.dumps(rows_a), encoding="utf-8")
    (cur / "results" / "grid" / "master.csv").write_bytes(("﻿" + MASTER.replace("0.862125", "0.862126")).encode())
    for mode in ("ref", "other"):
        plain = C.regen_compare(base, cur, mode, TOL_SMALL)
        full = C.regen_compare(base, cur, mode, TOL_SMALL, include_identical=True)
        assert "rows" not in plain and plain["counts"] == full["counts"]
        assert json.dumps(plain["findings"]) == json.dumps(full["findings"])
        rows = full["rows"]
        assert len([r for r in rows if r["kind"] == "leaf"]) == full["counts"]["json"]["leaves"] == 11
        assert len([r for r in rows if r["kind"] == "csv_cell"]) == full["counts"]["files"]["cells"] == 9
        assert all(r["gate"] is False and r["status"] in ("PASS", "REPORT") for r in rows if r["identical"])
        key = [(f["file"], f["path"], f["status"]) for f in full["findings"]]
        assert set(key) <= {(r["file"], r["path"], r["status"]) for r in rows}
    agg, tf = T.regen_tables(tmp_path / "out" / "regen", full["rows"], label="FAIL", mode="other")
    assert (agg.csv.name, tf.csv.name) == ("aggregates.csv", "table_files.csv")
    a = read_csv(agg.csv)
    assert len(a) == len([r for r in full["rows"] if r["section"] == "json"])
    by = {(r["파일"], r["JSON 경로"]): r for r in a}
    assert by[("results/a.json", "x/s")][V] == "FAIL"
    yv = by[("results/a.json", "y/v")]
    assert (yv[V], yv[T.DIFF]) == ("REPORT", "0.0005") and yv[T.TOL].startswith("abs 0.001 (넘으면 FAIL")
    assert by[("results/a.json", "x/v")][V] == "PASS" and by[("results/a.json", "x/v")][T.DIFF] == "0"
    assert by[("results/a.json", "x/p")][V] == "REPORT"                        # 보고 규칙은 같아도 REPORT
    assert by[("results/opt/metrics.json", "auc")][V] == "REPORT"
    f = read_csv(tf.csv)
    cell = next(r for r in f if r[V] == "FAIL")
    assert (cell["위치"], cell["행"], cell["열"], cell[T.DIFF]) == ("r1c2", "x", "top1", "0.000001")
    assert next(r for r in f if r["위치"] == "r0c1")["행"] == "(머리글)"
    assert [r[V] for r in f if r["파일"] == "results/det/table.md"] == ["REPORT"]
    assert read_md(agg.md)[1] == agg.count_line == T.count_line(tally(a))
    assert read_md(tf.md)[1] == T.count_line(tally(f))


def test_regen_table_collapses_skipped_and_stale_files(tmp_path):
    base, cur = _regen_trees(tmp_path)
    rows = C.regen_compare(base, cur, "ref", TOL_SMALL, include_identical=True)["rows"]
    skipped = {"S1.bank": {"reason": "측정 불가 — 체크포인트 없음", "outputs": ["results/opt/metrics.json"]},
               "S3.det": {"reason": "측정 불가 — 체크포인트 없음", "outputs": ["results/det/table.md"]}}
    stale = {"results/a.json": "S3.a: 다시 쓰이지 않음"}
    agg, tf = T.regen_tables(tmp_path / "out", rows, skipped=skipped, stale=stale)
    a, f = read_csv(agg.csv), read_csv(tf.csv)
    assert [(r["파일"], r["JSON 경로"], r[V]) for r in a] == [
        ("results/a.json", T.FILE_ROW, "FAIL"), ("results/opt/metrics.json", T.FILE_ROW, "N/A"),
        ("results/det/config_used.json", T.FILE_ROW, "N/A")]                   # 같은 폴더의 파일도 그 단계 산출물
    assert "다시 쓰이지 않음" in a[0][T.NOTE] and a[1][T.NOTE].startswith("S1.bank 건너뜀")
    assert a[0][T.COMMITTED] == a[0][T.NEW] == ""                              # 기준본 값을 새 값처럼 적지 않는다
    assert [r[V] for r in f if r["파일"] == "results/det/table.md"] == ["N/A"]
    assert {r[V] for r in f if r["파일"] == "results/grid/master.csv"} == {"PASS"}
    assert agg.count_line == "PASS 0 · FAIL 1 · REPORT 0 · N/A 2"


def test_file_overrides_maps_only_step_outputs():
    files = ["results/a.json", "results/b.json", "results/det/x.json", "results/det/y.md", "results/other/z.json"]
    out = T.file_overrides(files, {"S": {"reason": "r", "outputs": ["results/a.json", "results/det/y.md"]}},
                           {"results/b.json": "S2: 파일 없음", "results/nope.json": "무시"})
    assert out == {"results/a.json": ("N/A", "S 건너뜀: r"), "results/det/x.json": ("N/A", "S 건너뜀: r"),
                   "results/det/y.md": ("N/A", "S 건너뜀: r"), "results/b.json": ("FAIL", "S2: 파일 없음")}
    assert T.file_overrides(files) == {}


def test_regen_hook_helper_on_repository(tmp_path):
    """저장소 results/를 기준본·새 값 둘 다로(읽기만): ckpt 단계를 건너뛰면 그 산출물 6개만 N/A 한 행씩."""
    tol = C.load_yaml(DOCKER / "tolerances.yaml")
    rep = common.Report("regen", tmp_path)
    ckpt = [sid for sid in CLI.REGEN_ORDER if P.STEP_BY_ID[sid]["needs"] or P.STEP_BY_ID[sid]["needs_any"]]
    for sid in ckpt:
        rep.add_check(sid, "N/A", "측정 불가 — 사전조건 미충족(시험)")
    rep.add_check("S3.retrieval: 갱신 results/retrieval_eval.json", "FAIL", "다시 쓰이지 않음")
    agg, tf = CLI._regen_tables(tmp_path / "regen", rep, C, P, REPO, REPO, "ref", tol, "FAIL")
    a, f = read_csv(agg.csv), read_csv(tf.csv)
    na = [r for r in a + f if r[V] == "N/A"]
    assert sorted(r["파일"] for r in na) == sorted(CKPT_FILES)
    assert all(T.FILE_ROW in (r.get("JSON 경로"), r.get("위치")) for r in na)
    assert [r[V] for r in a if r["파일"] == "results/retrieval_eval.json"] == ["FAIL"]
    rest = [r for r in a if r["파일"] not in CKPT_FILES and r["파일"] != "results/retrieval_eval.json"]
    assert rest and {r[V] for r in rest} <= {"PASS", "REPORT"}
    cells = [r for r in f if r["위치"].startswith("r")]
    assert cells and {r[V] for r in cells} == {"PASS"}
    assert tally(a)["N/A"] == 5 and tally(f)["N/A"] == 1


# ---------------------------------------------------------------- train → trained_runs

def _metrics(root: Path, exp: str, obj: dict) -> None:
    p = root / "results" / exp / "metrics.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")


def test_train_table_reports_without_judging(tmp_path):
    base, wd = tmp_path / "bl", tmp_path / "wd"
    _metrics(base, "e1", {"top1": 0.5, "epochs": 40, "variant": {"apt": "reciprocal"}, "old": 1})
    _metrics(wd, "e1", {"top1": 0.625, "epochs": 40, "variant": {"apt": "reciprocal"}, "new_key": None})
    _metrics(wd, "e2", {"top1": 0.1})                                     # 기준본에 없는 런
    (t,) = T.train_tables(tmp_path / "out" / "train", base, wd, {"trained": ["e1", "e2", "e3", "e1"]})
    assert t.csv.name == "trained_runs.csv"
    rows = read_csv(t.csv)
    by = {(r["런"], r["지표"]): r for r in rows}
    assert (by[("e1", "top1")][T.DIFF], by[("e1", "top1")][V]) == ("0.125", "REPORT")
    assert by[("e1", "epochs")][T.DIFF] == "0" and by[("e1", "variant.apt")][T.COMMITTED] == "reciprocal"
    assert by[("e1", "old")][V] == by[("e1", "new_key")][V] == "N/A" and by[("e1", "new_key")][T.NEW] == "null"
    assert by[("e2", "top1")][V] == "N/A" and by[("e2", "top1")][T.COMMITTED] == ""
    assert by[("e3", "(metrics.json)")][V] == "N/A"                       # 새 metrics.json 없음
    assert t.count_line == "PASS 0 · FAIL 0 · REPORT 3 · N/A 4" and read_md(t.md)[1] == t.count_line


def test_train_table_empty_and_missing_baseline(tmp_path):
    (t,) = T.train_tables(tmp_path / "o1", tmp_path, tmp_path, {"trained": []})
    assert read_csv(t.csv) == [] and t.count_line == "PASS 0 · FAIL 0 · REPORT 0 · N/A 0"
    _metrics(tmp_path / "wd", "e1", {"top1": 0.5})
    (t,) = T.train_tables(tmp_path / "o2", None, tmp_path / "wd", {"trained": ["e1"]})
    rows = read_csv(t.csv)
    assert [(r["지표"], r[V]) for r in rows] == [("top1", "N/A")] and "기준본 위치 없음" in rows[0][T.NOTE]
    assert T.train_tables(tmp_path / "o3", None, tmp_path, None)[0].count_line == "PASS 0 · FAIL 0 · REPORT 0 · N/A 0"


def test_train_table_baseline_location(tmp_path, monkeypatch):
    wd = tmp_path / "wd"
    assert CLI._train_table_baseline(argparse.Namespace(baseline=None), wd) == wd / CLI.HOST_TRAIN_BASELINE_REL
    assert CLI._train_table_baseline(argparse.Namespace(baseline=str(tmp_path / "b")), wd) == (tmp_path / "b").resolve()
    monkeypatch.setenv("NMFC_IN_CONTAINER", "1")
    assert CLI._train_table_baseline(argparse.Namespace(baseline=None), wd) is None      # $NMFC_BASELINE 없음
    monkeypatch.setenv("NMFC_BASELINE", str(tmp_path / "opt"))
    assert CLI._train_table_baseline(argparse.Namespace(baseline=None), wd) == tmp_path / "opt"


def test_train_hook_writes_trained_runs_and_reproduce_does_not(tmp_path, monkeypatch):
    """학습은 흉내(run_pipeline 대체): train은 <out>/train/trained_runs를 쓰고, reproduce는 표를 쓰지 않는다."""
    wd = tmp_path / "wd"
    _metrics(wd, "syn_x", {"test_top1": 0.5, "epochs": 10})

    def fake_run(root, **kw):
        _metrics(Path(root), "syn_x", {"test_top1": 0.75, "epochs": 10})
        st = {"started": "x", "started_ns": time.time_ns(), "trained": ["syn_x"], "items": [], "pass": True,
              "exit_code": 0}
        (Path(root) / CLI.STATUS_REL).write_text(json.dumps(st), encoding="utf-8")
        return 0

    monkeypatch.setattr(P, "run_pipeline", fake_run)
    out = tmp_path / "out"
    assert CLI.main(["train", "--exp", "syn_x", "--workdir", str(wd), "--out", str(out)]) == 0
    rows = read_csv(out / "train" / "trained_runs.csv")
    assert [(r["런"], r["지표"], r[T.COMMITTED], r[T.NEW], r[T.DIFF], r[V]) for r in rows] == [
        ("syn_x", "test_top1", "0.5", "0.75", "0.25", "REPORT"), ("syn_x", "epochs", "10", "10", "0", "REPORT")]
    d = json.loads((out / "train_report.json").read_text(encoding="utf-8"))
    assert not [c for c in d["checks"] if c["name"] == "결과표"] and d["pass"] is True
    out2 = tmp_path / "out2"
    assert CLI.main(["reproduce", "--workdir", str(wd), "--out", str(out2)]) == 0
    assert not list(out2.rglob("trained_runs.*"))


# ---------------------------------------------------------------- compare → compare_table·gates

def _res() -> dict:
    return {
        "gates": [
            {"kind": "exact", "exp": "c100v_shrb_open", "metric": "label_noise_stats.n_flipped", "ref": 0, "new": 0,
             "status": "PASS", "detail": "런별 정확 일치(기준본 값)"},
            {"kind": "synthetic", "exp": "syn_moons_nmfc", "metric": "test_top1", "ref": 0.5, "new": 0.53,
             "delta": 0.030000000000000027, "threshold": 0.01, "status": "FAIL", "detail": "허용오차 ±0.01"},
            {"kind": "single_run", "family": "core", "group": "shrb", "exp": "c100v_shrb_s0", "seed": "0",
             "metric": "top1", "ref": 0.84, "new": 0.845, "delta": 0.005, "s": 0.002, "threshold": 0.0184,
             "m_run": 5, "df": 4, "status": "PASS", "detail": "단일 런"},
            {"kind": "mean", "family": "core", "group": "shrb", "metric": "map", "ref": 0.7, "new": 0.71,
             "delta": 0.01, "s": 0.003, "threshold": 0.0128, "n": 5, "m": 20, "status": "PASS", "detail": "평균"},
            {"kind": "sign", "family": "core", "group": "shrb_vs_softmax/map", "metric": "map", "ref": "5/5",
             "new": "5/5", "ratio": 4.2, "involved": ["c100v_shrb_s0"], "status": "PASS", "detail": "부호 5/5 유지"}],
        "reports": [
            {"kind": "sd_multiple", "family": "core", "group": "shrb", "exp": "c100v_shrb_s0", "metric": "top1",
             "delta": 0.005, "s": 0.002, "threshold": 0.0084, "status": "REPORT", "detail": "런별 4.2·SD 이내(보고만)"},
            {"kind": "single_seed", "exp": "cifar100_deepmfa", "metric": "top1", "ref": 0.83, "new": 0.86,
             "delta": 0.03, "threshold": 0.02, "status": "WARN", "detail": "분산 미상 단일 시드"},
            {"kind": "sign", "family": "ood", "group": "shrb_vs_mhrb/near_auc", "metric": "near_auc", "status": "N/A",
             "detail": "값 미평가"},
            {"kind": "file_change", "path": "results/syn_moons_nmfc/metrics.json", "change": "changed",
             "category": "trained", "status": "REPORT", "detail": "changed (trained)"},
            {"kind": "file_change", "path": "results/c100v_mhrb/metrics.json", "change": "changed",
             "category": "unexpected", "status": "WARN", "detail": "changed (unexpected)"}],
        "unevaluated": [
            {"exp": "c100v_shrb_s1", "metric": "map", "source": "results/seed_robustness.json",
             "reason": "results/seed_robustness.json가 기준본과 내용이 같음"},
            {"family": "core", "group": "shrb_vs_mhrb/map", "metric": "map", "gate": "sign", "reason": "값 미평가"}],
        "pass": False, "exit_code": 1, "precondition": {"ok": True, "reasons": []},
        "trained": ["syn_moons_nmfc", "c100v_shrb_s0"], "summary": {"exit_code": 1, "reason": "게이트 실패"},
        "runs": [{"state": "ok"}]}


def test_compare_tables_rows_and_verdicts(tmp_path):
    ct, gt = T.compare_tables(tmp_path / "out" / "compare", _res())
    assert (ct.csv.name, gt.csv.name) == ("compare_table.csv", "gates.csv")
    rows = read_csv(ct.csv)
    assert [(r["구분"], r["종류"], r[V]) for r in rows] == [
        ("게이트", "exact", "PASS"), ("게이트", "synthetic", "FAIL"), ("게이트", "single_run", "PASS"),
        ("게이트", "mean", "PASS"), ("게이트", "sign", "PASS"), ("미평가", "", "N/A"), ("미평가", "sign", "N/A"),
        ("보고", "sd_multiple", "REPORT"), ("보고", "single_seed", "REPORT"), ("보고", "sign", "N/A")]
    assert ct.count_line == "PASS 4 · FAIL 1 · REPORT 2 · N/A 3" and read_md(ct.md)[1] == ct.count_line
    syn, mean, sign = rows[1], rows[3], rows[4]
    assert (syn[T.COMMITTED], syn[T.NEW], syn[T.DIFF], syn[T.TOL]) == ("0.5", "0.53", "0.03", "|차이| ≤ 0.01")
    assert (mean["실험"], mean[T.TOL]) == ("core/shrb", "|평균 차이| ≤ 0.0128") and "n=5" in mean[T.NOTE]
    assert (sign["실험"], sign["지표"], sign[T.DIFF]) == ("core/shrb_vs_softmax", "map", "")
    sd = rows[7]                                                     # 같은 런·지표의 single_run 값을 빌린다
    assert (sd[T.COMMITTED], sd[T.NEW], sd[T.DIFF]) == ("0.84", "0.845", "0.005") and "|차이|=0.005" in sd[T.NOTE]
    assert rows[6]["실험"] == "core/shrb_vs_mhrb" and rows[8][T.TOL] == "(보고 기준) |차이| ≤ 0.02"
    assert not [r for r in rows if "metrics.json" in r["지표"]]          # file_change는 표에 넣지 않는다
    assert any("file_change) 2건" in line and "WARN 1건" in line for line in read_md(ct.md))
    g = read_csv(gt.csv)
    assert [r["게이트"] for r in g] == [
        "exact: c100v_shrb_open/label_noise_stats.n_flipped", "synthetic: syn_moons_nmfc/test_top1",
        "single_run: c100v_shrb_s0/top1", "mean: core/shrb/map", "sign: core/shrb_vs_softmax/map",
        "미평가: c100v_shrb_s1/map", "sign: core/shrb_vs_mhrb/map"]
    assert gt.count_line == "PASS 4 · FAIL 1 · REPORT 0 · N/A 2"
    assert (g[0]["통계량"], g[0]["값"], g[0][T.TOL]) == ("값 일치", "0", "= 0")
    assert (g[1]["값"], g[1][T.TOL], g[1][V]) == ("0.030000000000000027", "≤ 0.01", "FAIL")
    assert "5시드" in g[3]["통계량"] and (g[4]["값"], g[4][T.TOL]) == ("5/5", "= 5/5")


def test_compare_tables_precondition_failure_is_not_clean(tmp_path):
    res = {"gates": [], "reports": [], "unevaluated": [], "pass": False, "exit_code": 2,
           "precondition": {"ok": False, "reasons": ["a: metrics.json mtime이 실행 시작보다 이전임", "b: 없음"]},
           "summary": {"exit_code": 2, "reason": "전제조건 불충족"}}
    ct, gt = T.compare_tables(tmp_path, res)
    for t in (ct, gt):
        rows = read_csv(t.csv)
        assert [r[V] for r in rows] == ["FAIL", "FAIL"] and "mtime" in rows[0][T.NOTE]
        assert t.count_line == "PASS 0 · FAIL 2 · REPORT 0 · N/A 0"
    assert any(line.startswith("- 전제조건: 불충족") for line in read_md(ct.md))


def test_compare_tables_empty_result(tmp_path):
    ct, gt = T.compare_tables(tmp_path, {})
    assert read_csv(ct.csv) == [] == read_csv(gt.csv)
    assert ct.count_line == gt.count_line == "PASS 0 · FAIL 0 · REPORT 0 · N/A 0"


def _trained_workdir(tmp_path: Path, exp: str = "syn_moons_softmax") -> Path:
    """합성 1런을 '학습한' 작업 트리(test_cli와 같은 구성, 저장소 metrics.json은 읽기만)."""
    wd = tmp_path / "wd"
    ref = REPO / "results" / exp / "metrics.json"
    m = json.loads(ref.read_text(encoding="utf-8"))
    m["silhouette"] = m["silhouette"] - 0.001
    _metrics(wd, exp, m)
    (wd / CLI.HOST_TRAIN_BASELINE_REL / "results" / exp).mkdir(parents=True)
    shutil.copyfile(ref, wd / CLI.HOST_TRAIN_BASELINE_REL / "results" / exp / "metrics.json")
    t = time.time_ns() - 60 * 10**9
    ok = {"started": "2026-11-10T10:00:00+09:00", "started_ns": t, "finished": "x", "pass": True, "exit_code": 0,
          "args": {"exp": exp, "force": True}, "trained": [exp],
          "items": [{"id": f"T1:{exp}", "kind": "train", "required": True, "exit": 0, "ok": True, "expected": [exp],
                     "cached": [], "trained": [exp], "started_ns": t, "artifacts": {exp: P.run_artifacts(wd, exp, None)}}]}
    (wd / "runs" / "pipeline_status.json").write_text(json.dumps(ok), encoding="utf-8")
    return wd


def test_compare_hook_writes_tables(tmp_path, capsys):
    wd = _trained_workdir(tmp_path)
    out = tmp_path / "out"
    assert CLI.main(["compare", "--workdir", str(wd), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    res = json.loads((out / "compare_result.json").read_text(encoding="utf-8"))
    g = read_csv(out / "compare" / "gates.csv")
    assert len(g) == len(res["gates"]) >= 2 and {r[V] for r in g} == {"PASS"}
    line = f"PASS {len(g)} · FAIL 0 · REPORT 0 · N/A 0"
    assert read_md(out / "compare" / "gates.md")[1] == line and line in printed
    assert read_md(out / "compare" / "compare_table.md")[1].startswith(f"PASS {len(g)} · FAIL 0 · ")
    d = json.loads((out / "compare_report.json").read_text(encoding="utf-8"))
    assert not [c for c in d["checks"] if c["name"] == "결과표"]
    assert [t["name"] for t in d["env"]["result_tables"]["tables"]] == ["compare_table", "gates"]


def test_table_failure_keeps_exit_code_and_leaves_warn(tmp_path, monkeypatch, capsys):
    wd = _trained_workdir(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("표 작성 실패 흉내")

    monkeypatch.setattr(T, "compare_tables", boom)
    out = tmp_path / "out"
    assert CLI.main(["compare", "--workdir", str(wd), "--out", str(out)]) == 0
    d = json.loads((out / "compare_report.json").read_text(encoding="utf-8"))
    warn = [c for c in d["checks"] if c["name"] == "결과표"]
    assert len(warn) == 1 and warn[0]["status"] == "WARN" and "표 작성 실패 흉내" in warn[0]["detail"]
    assert d["pass"] is True and d["label"] == "PASS" and "result_tables" not in d["env"]
    assert "결과표 쓰기 실패" in capsys.readouterr().out
    assert not (out / "compare").exists()


def test_write_tables_helper(tmp_path, capsys):
    rep = common.Report("verify", tmp_path)
    a = argparse.Namespace(out=str(tmp_path / "o"))
    CLI._write_tables(rep, a, "verify", lambda d: [T.write_table(d, "x", "x", ("a", V), [{"a": 1, V: "PASS"}])])
    assert (tmp_path / "o" / "verify" / "x.csv").is_file() and rep.checks == []
    assert "결과표 x: PASS 1 · FAIL 0 · REPORT 0 · N/A 0" in capsys.readouterr().out
    CLI._write_tables(rep, a, "verify", lambda d: [T.write_table(d, "y", "y", ("a",), [])])   # 판정 열 없음
    assert [c["status"] for c in rep.checks] == ["WARN"] and rep.passed
