"""nmfc_tables — nmfc 명령의 결과표(CSV·Markdown).

nmfc_cli가 verify·regen·train·compare를 마치기 직전(호스트 일회용 사본을 지우기 전)에 이 모듈을 지연 import해서
<out>/<명령>/<표 이름>.csv·.md를 쓴다. 최상위 import는 표준 라이브러리뿐이다(nmfc_cli의 --help·manifest 경로가
torch·numpy·matplotlib를 끌어오지 않게). 비교는 하지 않는다: 판정은 nmfc_compare가 이미 낸 결과를 옮겨 적는다.

명령별 표
  verify   theory_claims   results/theory_audit.json — checks 원소마다 한 행, 통과 수(n_passed/n_total) 한 행,
                           그 밖의 최상위 키(config·R 등)마다 한 행
  regen    aggregates      tolerances.yaml json 절 파일 — 비교한 모든 리프(값이 같은 리프 포함)와 구조 차이
           table_files     tolerances.yaml files 절 — CSV는 셀마다, PNG·md는 파일마다 한 행
  train    trained_runs    이번 train 실행에서 학습한 런 — metrics.json의 리프마다 한 행(train은 판정하지 않는다)
  compare  compare_table   전제조건·게이트·미평가·보고 항목(실험 × 지표)
           gates           전제조건·게이트·미평가 게이트

열
  커밋 값(results/)  기준본의 값(컨테이너 /opt/nmfc/baseline, 호스트는 실행 전 스냅샷 = 저장소의 results/)
  새 값              이번 실행이 쓴 값
  차이               새 값 − 커밋 값(부호 있음). 두 값이 모두 수일 때만 적는다
  허용범위           판정 규칙(tolerances.yaml, verify는 float_platform)
  판정               표마다 한 열. 값은 PASS · FAIL · REPORT · N/A뿐이다(to_verdict).
                     PASS·FAIL = 판정 결과, REPORT = 판정 없이 보고(보고 규칙·WARN·INFO), N/A = 평가하지 않음
                     (건너뜀·측정 불가·값 없음). 판정하지 않은 행은 PASS로 적지 않는다.
  비고               비교기가 남긴 사실 문구

CSV는 utf-8-sig(csv 모듈, 줄끝 CRLF), Markdown은 UTF-8·LF이고 열이 같다. Markdown은 제목 줄과 판정 값별 건수 줄
(예: PASS 3102 · FAIL 0 · REPORT 4 · N/A 5)로 시작한다. 결과표는 종료코드에 영향이 없다(쓰기 실패는 nmfc_cli가
WARN 검사로 남긴다).
"""
from __future__ import annotations

import csv
import dataclasses
import decimal
import json
import math
import os
import posixpath
import re
from pathlib import Path

VERDICTS: tuple[str, ...] = ("PASS", "FAIL", "REPORT", "N/A")
VERDICT = "판정"
COMMITTED = "커밋 값(results/)"
NEW = "새 값"
DIFF = "차이"
TOL = "허용범위"
NOTE = "비고"
LEGEND = ("차이 = 새 값 − 커밋 값(두 값이 모두 수일 때). 판정: PASS·FAIL = 판정 결과, REPORT = 판정 없이 보고, "
          "N/A = 평가하지 않음.")

THEORY_AUDIT_REL = "results/theory_audit.json"
FP_REL, FP_ABS = 1e-9, 1e-12          # nmfc_compare.compare_float_platform의 기본값(verify의 float_platform)
FP_EXACT_PATHS = ("checks/*/name", "checks/*/passed", "n_passed", "n_total")   # nmfc_compare.FP_EXACT_PATHS
FP_TOL_TEXT = f"isclose(rel={FP_REL:g}, abs={FP_ABS:g})"

THEORY_COLUMNS = ("구분", "항목", COMMITTED, NEW, DIFF, TOL, VERDICT, NOTE)
AGGREGATE_COLUMNS = ("파일", "JSON 경로", COMMITTED, NEW, DIFF, TOL, VERDICT, NOTE)
TABLE_FILE_COLUMNS = ("파일", "위치", "행", "열", COMMITTED, NEW, DIFF, TOL, VERDICT, NOTE)
TRAINED_COLUMNS = ("런", "지표", COMMITTED, NEW, DIFF, VERDICT, NOTE)
COMPARE_COLUMNS = ("구분", "종류", "실험", "지표", COMMITTED, NEW, DIFF, TOL, VERDICT, NOTE)
GATE_COLUMNS = ("게이트", "통계량", "값", TOL, VERDICT, NOTE)

FILE_ROW = "(파일 전체)"
MAX_TEXT = 300                         # 컨테이너 값·긴 비고를 자르는 길이(CSV·MD 공통)
_MISSING = object()
_CELL_RE = re.compile(r"r(\d+)c(\d+)")
_ROW_RE = re.compile(r"r(\d+)")
_NA_STATUS = frozenset({"N/A", "NA", "SKIP", "SKIPPED", "UNEVALUATED", "NOT EVALUATED", "NOT_EVALUATED"})


@dataclasses.dataclass(frozen=True)
class Table:
    """write_table이 쓴 표 하나."""
    name: str
    csv: Path
    md: Path
    counts: dict
    count_line: str


# ============================================================================ 공통

def to_verdict(status) -> str:
    """비교기 상태 → 판정 열 값. PASS·FAIL은 그대로, 평가하지 않음(N/A·건너뜀·None·빈 값)은 N/A,
    그 밖(REPORT·WARN·INFO·알 수 없는 값)은 REPORT. 판정하지 않은 것을 PASS로 바꾸지 않는다."""
    if status is None:
        return "N/A"
    s = str(status).strip().upper()
    if s in ("PASS", "FAIL"):
        return s
    if not s or s in _NA_STATUS:
        return "N/A"
    return "REPORT"


def count_line(counts: dict) -> str:
    """판정 값별 건수 한 줄. 네 값을 늘 같은 순서로(0건 포함) 적는다."""
    return " · ".join(f"{v} {int((counts or {}).get(v, 0))}" for v in VERDICTS)


def _clip(s: str, n: int | None = MAX_TEXT) -> str:
    return s if n is None or len(s) <= n else s[:n] + "…"


def fmt_value(v, *, clip: int | None = MAX_TEXT) -> str:
    """JSON 값의 표 표기: null·true·false, float는 repr(왕복 가능한 최단 표기), 문자열은 그대로, 컨테이너는 JSON.
    값이 없는 쪽(_MISSING)은 빈 칸."""
    if v is _MISSING:
        return ""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "Infinity" if v > 0 else "-Infinity"
        return repr(v)
    if isinstance(v, str):
        return v
    try:
        s = json.dumps(v, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        s = str(v)
    return _clip(s, clip)


def _num(x):
    """int·float(bool 제외)면 그대로, 아니면 None."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return x


def _shown_decimal(x):
    """수 → 표에 적는 표기(float는 repr)와 같은 십진수. NaN·무한대는 None."""
    try:
        d = decimal.Decimal(repr(x)) if isinstance(x, float) else decimal.Decimal(x)
    except (decimal.InvalidOperation, ValueError, TypeError):
        return None
    return d if d.is_finite() else None


def signed_diff(ref, new):
    """새 값 − 커밋 값. 두 값이 모두 유한한 수(bool 제외)일 때만, 아니면 None.
    int끼리는 int, 그 밖은 표에 적힌 두 표기의 십진 차이(Decimal)다(표의 두 값을 손으로 뺀 값과 같다)."""
    a, b = _num(ref), _num(new)
    if a is None or b is None:
        return None
    if isinstance(a, int) and isinstance(b, int):
        return b - a
    da, db = _shown_decimal(a), _shown_decimal(b)
    return None if da is None or db is None else db - da


def _fmt_decimal(d: decimal.Decimal) -> str:
    """십진 차이 표기: 0은 '0', 지수 −6~15는 고정소수점, 그 밖은 지수 표기(소문자 e)."""
    if d == 0:
        return "0"
    d = d.normalize()
    return format(d, "f") if -6 <= d.adjusted() < 16 else format(d, "E").lower()


def fmt_diff(d) -> str:
    if d is None:
        return ""
    return _fmt_decimal(d) if isinstance(d, decimal.Decimal) else fmt_value(d)


def _decimal(s):
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        d = decimal.Decimal(s.strip())
    except (decimal.InvalidOperation, ValueError):
        return None
    return d if d.is_finite() else None


def text_number_diff(ref, new) -> str:
    """두 문자열(CSV 셀)이 모두 유한한 십진수면 새 값 − 커밋 값(십진 계산이라 표기 그대로 뺀다), 아니면 ''."""
    a, b = _decimal(ref), _decimal(new)
    if a is None or b is None:
        return ""
    return _fmt_decimal(b - a)


def tol_label(text) -> str:
    """허용범위 표기. on_diff warn 규칙('…, WARN')은 판정 열 어휘에 맞춰 풀어 쓴다."""
    t = "" if text is None else str(text)
    if t.endswith(", WARN"):
        return t[: -len(", WARN")] + " (넘으면 FAIL, 안이지만 값이 다르면 REPORT)"
    return t


def _read_json(path):
    """(값, 오류 문구). 없거나 못 읽으면 값은 None. 비교기와 같게 utf-8-sig로 읽는다."""
    if path is None:
        return None, "위치 없음"
    p = Path(path)
    if not p.is_file():
        return None, f"파일 없음({p.as_posix()})"
    try:
        return json.loads(p.read_text(encoding="utf-8-sig")), None
    except (OSError, ValueError) as e:
        return None, f"읽기 실패: {type(e).__name__}: {e}"


def _one_line(s) -> str:
    return " ".join(str(s).split())


def _md(s) -> str:
    """표 칸 이스케이프(nmfc_common Report.to_markdown과 같은 규칙에 '*' 추가)."""
    s = "" if s is None else str(s)
    s = s.replace("\\", "\\\\").replace("|", "\\|").replace("*", "\\*")
    return s.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")


def _cell(v) -> str:
    if v is None:
        return ""
    return v if isinstance(v, str) else fmt_value(v)


def write_table(dest, name: str, title: str, columns, rows, notes=()) -> Table:
    """<dest>/<name>.csv(utf-8-sig)와 .md(같은 열)를 쓴다. rows는 {열 이름: 값} dict 목록.

    열에 '판정'이 정확히 하나여야 하고, 행의 판정은 VERDICTS 중 하나여야 한다(아니면 ValueError —
    nmfc_cli는 이것을 WARN 검사로 남긴다). Markdown은 제목 줄, 건수 줄, 빈 줄, 범례, notes, 표 순서다."""
    columns = [str(c) for c in columns]
    if columns.count(VERDICT) != 1 or len(set(columns)) != len(columns):
        raise ValueError(f"표 {name}: 열 이름이 겹치거나 '{VERDICT}' 열이 하나가 아니다: {columns}")
    if not re.fullmatch(r"[A-Za-z0-9_]+", str(name)):
        raise ValueError(f"표 이름은 영문·숫자·밑줄만: {name!r}")
    counts = {v: 0 for v in VERDICTS}
    body: list[list[str]] = []
    for r in rows or ():
        extra = set(r) - set(columns)
        if extra:
            raise ValueError(f"표 {name}: 열에 없는 키 {sorted(extra)}")
        v = r.get(VERDICT)
        if v not in VERDICTS:
            raise ValueError(f"표 {name}: 판정 값은 {' · '.join(VERDICTS)} 중 하나여야 한다: {v!r}")
        counts[v] += 1
        body.append([_cell(r.get(c)) for c in columns])
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    csv_path, md_path = dest / f"{name}.csv", dest / f"{name}.md"
    line = count_line(counts)

    tmp = csv_path.with_name(f"{csv_path.name}.tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh, lineterminator="\r\n")
        w.writerow(columns)
        w.writerows(body)
    os.replace(tmp, csv_path)

    md = [f"# {_one_line(title)}", line, "", LEGEND, ""]
    notes = [_one_line(n).replace("*", "\\*") for n in notes or () if n]
    if notes:
        md += [f"- {n}" for n in notes] + [""]
    md.append("| " + " | ".join(_md(c) for c in columns) + " |")
    md.append("|" + "---|" * len(columns))
    md += ["| " + " | ".join(_md(x) for x in row) + " |" for row in body]
    if not body:
        md += ["", "(행 없음)"]
    tmp = md_path.with_name(f"{md_path.name}.tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(md) + "\n")
    os.replace(tmp, md_path)
    return Table(str(name), csv_path, md_path, counts, line)


# ============================================================================ verify

def _under(path, prefix: str) -> bool:
    p = str(path)
    return p == prefix or p.startswith(prefix + "/")


def _finding_diffs(f: dict):
    """finding 하나의 부호 있는 차이들: 수 리프는 새 − 커밋(float), 문자열은 속의 수마다(개수가 같을 때, 십진 계산)."""
    d = signed_diff(f.get("ref"), f.get("new"))
    if d is not None:
        yield d
    data = f.get("data") if isinstance(f.get("data"), dict) else {}
    na, nb = data.get("numbers_ref"), data.get("numbers_new")
    if isinstance(na, list) and isinstance(nb, list) and len(na) == len(nb):
        for x, y in zip(na, nb):
            a, b = _decimal(str(x)), _decimal(str(y))
            if a is not None and b is not None:
                yield b - a


def _largest_diff(fs, *, identical: bool) -> str:
    """행에 속한 finding의 차이 중 절댓값이 가장 큰 것(부호 유지). 다른 리프가 없으면 '0'."""
    best = None
    for f in fs:
        for d in _finding_diffs(f):
            if best is None or abs(d) > abs(best):
                best = d
    if best is not None:
        return fmt_diff(best)
    return "0" if identical else ""


def _check_text(c) -> str:
    if c is _MISSING:
        return ""
    if isinstance(c, dict):
        detail = c.get("detail")
        return f"[passed={fmt_value(c.get('passed', _MISSING)) or '-'}] {'' if detail is None else fmt_value(detail)}".strip()
    return fmt_value(c)


def _ratio_text(d: dict) -> str:
    if "n_passed" not in d and "n_total" not in d:
        return ""
    return f"{fmt_value(d.get('n_passed', _MISSING)) or '-'}/{fmt_value(d.get('n_total', _MISSING)) or '-'}"


def theory_rows(ref, new, wf: dict | None = None, *, ref_err: str | None = None, new_err: str | None = None) -> list[dict]:
    """theory_audit.json 두 개(커밋 값·새 값)와 compare_written 판정 wf로 theory_claims 행을 만든다.

    판정: wf(class float_platform)의 data.findings는 값이 다른 리프만 담는다. 행에 속한 finding에 FAIL이 있으면 FAIL,
    없으면 PASS. 파일 단위 실패(새 파일 없음·다시 쓰이지 않음·기준본 없음·읽기 실패)면 모든 행이 FAIL이다.
    wf가 없으면(verify가 이 파일을 판정하지 않음) 두 값이 있는 행은 REPORT, 한쪽이 없으면 N/A다."""
    judged = isinstance(wf, dict)
    data = (wf.get("data") if judged and isinstance(wf.get("data"), dict) else None) or {}
    fl = [f for f in data.get("findings") or [] if isinstance(f, dict)]
    file_verdict = file_note = None
    if judged:
        st = to_verdict(wf.get("status"))
        if str(wf.get("class", "float_platform")) != "float_platform":
            file_verdict, file_note = st, str(wf.get("detail") or "")
        elif st == "FAIL" and (not data.get("fresh") or not any(f.get("status") == "FAIL" for f in fl)):
            file_verdict, file_note = "FAIL", str(wf.get("detail") or "파일 판정 FAIL")

    def verdict(fs, both: bool) -> str:
        if file_verdict is not None:
            return file_verdict
        if not judged:
            return "REPORT" if both else "N/A"
        if any(f.get("status") == "FAIL" for f in fs) or not both:
            return "FAIL"
        return "PASS"

    def note(fs) -> str:
        if file_note is not None:
            return file_note
        if not judged:
            return "판정 없음(verify가 이 파일을 판정하지 않음)"
        if not fs:
            return "값 동일"
        return _clip("; ".join(f"{f.get('path')}: {f.get('detail')}" for f in fs), 2 * MAX_TEXT)

    if ref is None and new is None:
        errs = [f"커밋 값: {ref_err}" if ref_err else "", f"새 값: {new_err}" if new_err else "", file_note or ""]
        return [{"구분": "파일", "항목": FILE_ROW, COMMITTED: "", NEW: "", DIFF: "", TOL: "",
                 VERDICT: file_verdict or ("FAIL" if judged else "N/A"), NOTE: "; ".join(e for e in errs if e)}]

    rd = ref if isinstance(ref, dict) else {}
    nd = new if isinstance(new, dict) else {}
    rc = rd.get("checks") if isinstance(rd.get("checks"), list) else []
    nc = nd.get("checks") if isinstance(nd.get("checks"), list) else []
    rows: list[dict] = []
    for i in range(max(len(rc), len(nc))):
        a = rc[i] if i < len(rc) else _MISSING
        b = nc[i] if i < len(nc) else _MISSING
        fs = [f for f in fl if _under(f.get("path"), f"checks/{i}") or str(f.get("path")) == "checks"]
        both = a is not _MISSING and b is not _MISSING
        name = next((str(c["name"]) for c in (a, b) if isinstance(c, dict) and c.get("name") is not None), "")
        rows.append({"구분": "검사", "항목": f"{i}: {name}" if name else str(i),
                     COMMITTED: _check_text(a), NEW: _check_text(b), DIFF: _largest_diff(fs, identical=both and not fs),
                     TOL: f"name·passed 정확 일치, detail 속 수 {FP_TOL_TEXT}", VERDICT: verdict(fs, both),
                     NOTE: note(fs)})
    if any(k in d for d in (rd, nd) for k in ("n_passed", "n_total")):
        fs = [f for f in fl if str(f.get("path")) in ("n_passed", "n_total")]
        both = all(k in d for d in (rd, nd) for k in ("n_passed", "n_total"))
        rows.append({"구분": "합계", "항목": "통과 수(n_passed/n_total)", COMMITTED: _ratio_text(rd),
                     NEW: _ratio_text(nd), DIFF: fmt_diff(signed_diff(rd.get("n_passed"), nd.get("n_passed"))),
                     TOL: "정확 일치", VERDICT: verdict(fs, both), NOTE: note(fs)})
    skip = {"n_passed", "n_total"}
    if any(isinstance(d.get("checks"), list) for d in (rd, nd)):
        skip.add("checks")                           # 위의 검사 행이 다룬다(리스트가 아니면 수치 행 하나로 적는다)
    keys = [k for k in rd if k not in skip] + [k for k in nd if k not in skip and k not in rd]
    for k in keys:
        a, b = rd.get(k, _MISSING), nd.get(k, _MISSING)
        fs = [f for f in fl if _under(f.get("path"), str(k))]
        both = a is not _MISSING and b is not _MISSING
        rows.append({"구분": "수치", "항목": str(k), COMMITTED: fmt_value(a, clip=None), NEW: fmt_value(b, clip=None),
                     DIFF: _largest_diff(fs, identical=both and not fs),
                     TOL: f"수 {FP_TOL_TEXT}, 문자열은 수를 뺀 텍스트 정확 일치", VERDICT: verdict(fs, both),
                     NOTE: note(fs)})
    return rows


def verify_tables(dest, baseline, current, written, rel: str = THEORY_AUDIT_REL) -> list[Table]:
    """verify → theory_claims. written = compare_written 결과 목록(파일마다 {path, class, status, detail, data})."""
    rel = str(rel).replace("\\", "/")
    wf = next((f for f in written or () if isinstance(f, dict)
               and str(f.get("path", "")).replace("\\", "/") == rel), None)
    ref, ref_err = _read_json(Path(baseline) / rel) if baseline is not None else (None, "기준본 위치 없음")
    new, new_err = _read_json(Path(current) / rel)
    rows = theory_rows(ref, new, wf, ref_err=ref_err, new_err=new_err)
    how = (f"파일 판정 {wf.get('status')}({wf.get('class')}): {wf.get('detail')}" if wf is not None else
           "파일 판정 없음: verify의 다시 쓴 파일(expected.yaml writes)에 이 파일이 없다")
    notes = [f"{rel} — 커밋 값 = 기준본 {baseline}, 새 값 = 이번 verify 작업 트리 {current}",
             f"허용범위 float_platform: 수와 문자열 속 수는 {FP_TOL_TEXT}, {', '.join(FP_EXACT_PATHS)}는 정확 일치",
             "검사 행의 차이 = detail 속 수 가운데 |차이|가 가장 큰 것, 수치 행은 그 키 아래 리프 가운데 가장 큰 것",
             how]
    return [write_table(dest, "theory_claims", f"verify 결과표 — theory_claims ({rel})", THEORY_COLUMNS, rows, notes)]


# ============================================================================ regen

def file_overrides(files, skipped: dict | None = None, stale: dict | None = None) -> dict:
    """파일 → (판정, 비고). 이 파일들은 리프·셀 행 대신 파일마다 한 행으로 적는다.

    stale = {저장소 상대 경로: 사유}: 단계는 돌았지만 다시 쓰이지 않은 파일 → FAIL.
    skipped = {단계 id: {"reason", "outputs"}}: 건너뛴 단계의 산출물 → N/A. 산출물은 outputs(단계 post의 fresh 경로)
    자체와, 그 경로의 폴더가 results/ 아래 하위 폴더(results/openset_detection 등)면 그 폴더 바로 아래 파일이다."""
    files = [str(f) for f in files]
    out: dict = {}
    for rel, why in (stale or {}).items():
        rel = str(rel).replace("\\", "/")
        if rel in files:
            out[rel] = ("FAIL", str(why))
    for sid, s in (skipped or {}).items():
        s = s if isinstance(s, dict) else {}
        reason = str(s.get("reason") or "")
        for o in s.get("outputs") or ():
            o = str(o).replace("\\", "/")
            d = posixpath.dirname(o)
            for f in files:
                if f not in out and (f == o or ("/" in d and posixpath.dirname(f) == d)):
                    out[f] = ("N/A", f"{sid} 건너뜀: {reason}" if reason else f"{sid} 건너뜀")
    return out


def _collapse(rows, override: dict, build, file_row) -> list[dict]:
    out, done = [], set()
    for r in rows:
        f = str(r.get("file") or "")
        if f in override:
            if f not in done:
                done.add(f)
                out.append(file_row(f, *override[f]))
            continue
        out.append(build(r))
    return out


def _aggregate_row(r: dict) -> dict:
    kind = r.get("kind") or ("leaf" if r.get("path") else "presence")
    ref, new = r.get("ref"), r.get("new")
    if kind == "leaf":
        cv, nv = fmt_value(ref), fmt_value(new)
        d = signed_diff(ref, new)                 # 행의 diff(float 뺄셈) 대신 두 표기의 십진 차이
    elif kind in ("structure", "order"):          # 없는 쪽은 None → 빈 칸
        cv, nv = ("" if ref is None else fmt_value(ref)), ("" if new is None else fmt_value(new))
        d = None
    else:
        cv = nv = ""
        d = None
    return {"파일": str(r.get("file") or ""), "JSON 경로": str(r.get("path") or "") or FILE_ROW,
            COMMITTED: cv, NEW: nv, DIFF: fmt_diff(d), TOL: tol_label(r.get("tol_text", r.get("check"))),
            VERDICT: to_verdict(r.get("status")), NOTE: str(r.get("detail") or "")}


def _csv_labels(rows) -> dict:
    """CSV 파일 → (열 머리글 {j: 이름}, 행 이름 {i: exp 열(없으면 첫 열) 값}). 커밋 값 쪽 셀을 쓴다."""
    grid: dict = {}
    for r in rows:
        m = _CELL_RE.fullmatch(str(r.get("path") or ""))
        if m:
            v = r.get("ref") if r.get("ref") is not None else r.get("new")
            grid.setdefault(str(r.get("file") or ""), {})[(int(m.group(1)), int(m.group(2)))] = v
    out = {}
    for f, cells in grid.items():
        header = {j: v for (i, j), v in cells.items() if i == 0}
        key = next((j for j, v in sorted(header.items()) if str(v) == "exp"), 0)
        out[f] = (header, {i: v for (i, j), v in cells.items() if j == key and i > 0})
    return out


def _table_file_row(r: dict, labels: dict) -> dict:
    f, path = str(r.get("file") or ""), str(r.get("path") or "")
    header, keys = labels.get(f, ({}, {}))
    row_l = col_l = cv = nv = dv = ""
    m, mr = _CELL_RE.fullmatch(path), _ROW_RE.fullmatch(path)
    if m:
        i, j = int(m.group(1)), int(m.group(2))
        row_l = "(머리글)" if i == 0 else _cell(keys.get(i))
        col_l = _cell(header.get(j))
        cv, nv = _cell(r.get("ref")), _cell(r.get("new"))
        dv = text_number_diff(r.get("ref"), r.get("new"))
    elif path == "(행 수)" or mr:                  # 행·열 수 차이
        if mr:
            row_l = "(머리글)" if int(mr.group(1)) == 0 else _cell(keys.get(int(mr.group(1))))
        cv, nv = fmt_value(r.get("ref")), fmt_value(r.get("new"))
        dv = fmt_diff(signed_diff(r.get("ref"), r.get("new")))
    note = str(r.get("detail") or "")
    data = r.get("data")
    if r.get("kind") == "png" and isinstance(data, dict) and not data.get("pixel_equal"):
        note += (f" — 크기 {data.get('size_a')}→{data.get('size_b')}, 바뀐 픽셀 비율 {data.get('changed_ratio')},"
                 f" 최대 채널 차 {data.get('max_channel_diff')}")
    return {"파일": f, "위치": path or FILE_ROW, "행": row_l, "열": col_l, COMMITTED: cv, NEW: nv, DIFF: dv,
            TOL: tol_label(r.get("tol_text", r.get("check"))), VERDICT: to_verdict(r.get("status")), NOTE: note}


def regen_tables(dest, rows, *, skipped: dict | None = None, stale: dict | None = None,
                 label: str | None = None, mode: str | None = None) -> list[Table]:
    """regen → aggregates(json 절)·table_files(files 절).

    rows = nmfc_compare.regen_compare(..., include_identical=True)["rows"](값이 같은 항목 포함). skipped·stale은
    file_overrides 참고: 건너뛴 단계의 파일은 N/A 한 행, 다시 쓰이지 않은 파일은 FAIL 한 행이 된다(기준본과 같은
    내용을 '새 값'처럼 적지 않으려고)."""
    rows = [r for r in rows or () if isinstance(r, dict)]
    override = file_overrides(dict.fromkeys(str(r.get("file") or "") for r in rows), skipped, stale)
    jrows = [r for r in rows if r.get("section", "json") == "json"]
    frows = [r for r in rows if r.get("section") == "files"]
    labels = _csv_labels(frows)
    agg = _collapse(jrows, override, _aggregate_row,
                    lambda f, v, why: {"파일": f, "JSON 경로": FILE_ROW, VERDICT: v, NOTE: why})
    tf = _collapse(frows, override, lambda r: _table_file_row(r, labels),
                   lambda f, v, why: {"파일": f, "위치": FILE_ROW, VERDICT: v, NOTE: why})
    head = []
    if label or mode:
        head.append(f"regen 라벨 {label or '-'} · 판정 모드 {mode or '-'}(ref = tolerances.yaml reference_device와 같은 장치)")
    if skipped:
        head.append("건너뛴 단계(그 단계의 파일은 파일마다 N/A 한 행): "
                    + "; ".join(f"{sid} — {(s or {}).get('reason') or ''}" for sid, s in skipped.items()))
    if stale:
        head.append("다시 쓰이지 않은 파일(파일마다 FAIL 한 행): " + "; ".join(f"{k} — {v}" for k, v in stale.items()))
    return [write_table(dest, "aggregates", "regen 결과표 — aggregates (tolerances.yaml json 절)", AGGREGATE_COLUMNS,
                        agg, head + ["행 = 비교한 모든 JSON 리프(값이 같은 리프 포함)와 구조 차이. 행 키로 짝지은 리스트는 "
                                     "경로에 행 이름을 쓴다(예: c100v_shrb/mAP, 복합 키는 exp:variant)."]),
            write_table(dest, "table_files", "regen 결과표 — table_files (tolerances.yaml files 절)", TABLE_FILE_COLUMNS,
                        tf, head + ["행 = CSV 셀마다(위치 r<행>c<열>, r0 = 머리글, 행 이름 = exp 열), PNG·md는 파일마다."])]


# ============================================================================ train

def _flatten(obj, prefix: str = "") -> dict:
    """중첩 dict·list를 '.'로 이은 키의 리프 dict로. 빈 컨테이너는 그대로 리프다."""
    if isinstance(obj, dict) and obj:
        items = [(str(k), v) for k, v in obj.items()]
    elif isinstance(obj, list) and obj:
        items = [(str(i), v) for i, v in enumerate(obj)]
    else:
        return {prefix: obj} if prefix else {}
    out: dict = {}
    for k, v in items:
        out.update(_flatten(v, f"{prefix}.{k}" if prefix else k))
    return out


def train_tables(dest, baseline, current, status) -> list[Table]:
    """train → trained_runs. status = 이번 실행의 runs/pipeline_status.json(trained = 이번에 학습한 런).
    커밋 값 = <baseline>/results/<런>/metrics.json, 새 값 = <current>/results/<런>/metrics.json.
    train은 판정하지 않는다: 두 값이 있으면 REPORT, 한쪽이 없으면 N/A."""
    st = status if isinstance(status, dict) else {}
    trained = list(dict.fromkeys(str(e) for e in st.get("trained") or []))
    rows: list[dict] = []
    for exp in trained:
        if baseline is not None:
            ref, rerr = _read_json(Path(baseline) / "results" / exp / "metrics.json")
        else:
            ref, rerr = None, "기준본 위치 없음"
        new, nerr = _read_json(Path(current) / "results" / exp / "metrics.json")
        rl = _flatten(ref) if isinstance(ref, dict) else {}
        nl = _flatten(new) if isinstance(new, dict) else {}
        if not rl and not nl:
            rows.append({"런": exp, "지표": "(metrics.json)", VERDICT: "N/A",
                         NOTE: f"커밋 값: {rerr or '리프 없음'}; 새 값: {nerr or '리프 없음'}"})
            continue
        for k in list(rl) + [k for k in nl if k not in rl]:
            a, b = rl.get(k, _MISSING), nl.get(k, _MISSING)
            both = a is not _MISSING and b is not _MISSING
            if both:
                why = ""
            elif a is _MISSING:
                why = f"커밋 값 없음 — {rerr}" if rerr else "기준본 metrics.json에 없는 키"
            else:
                why = f"새 값 없음 — {nerr}" if nerr else "새 metrics.json에 없는 키"
            rows.append({"런": exp, "지표": k, COMMITTED: fmt_value(a), NEW: fmt_value(b),
                         DIFF: fmt_diff(signed_diff(a, b)) if both else "", VERDICT: "REPORT" if both else "N/A",
                         NOTE: why})
    notes = [f"이번 train 실행에서 학습한 런 {len(trained)}개(runs/pipeline_status.json의 trained)",
             f"커밋 값 = {baseline}의 results/<런>/metrics.json(학습 전 기준본), 새 값 = {current}의 같은 파일",
             "train은 판정하지 않는다(두 값이 있으면 REPORT, 한쪽이 없으면 N/A). 판정은 nmfc compare가 한다."]
    return [write_table(dest, "trained_runs", "train 결과표 — trained_runs (이번 실행에서 학습한 런)",
                        TRAINED_COLUMNS, rows, notes)]


# ============================================================================ compare

def _who(e: dict) -> str:
    """실험 칸: exp, 없으면 family/group(부호 항목은 group 끝의 '/지표'를 뗀다)."""
    if e.get("exp"):
        return str(e["exp"])
    grp = "" if e.get("group") is None else str(e["group"])
    metric = str(e.get("metric") or "")
    if "sign" in (e.get("kind"), e.get("gate")) and metric and grp.endswith("/" + metric):
        grp = grp[: -len(metric) - 1]
    return "/".join(x for x in (str(e.get("family") or ""), grp) if x)


def _extras(e: dict) -> str:
    keys = ("n", "s", "m", "m_run", "df", "ratio", "source", "transition_reason")
    return ", ".join(f"{k}={fmt_value(e[k])}" for k in keys if e.get(k) is not None)


def _note_of(e: dict) -> str:
    base = str(e.get("detail") or e.get("reason") or "")
    ex = _extras(e)
    if e.get("kind") == "sd_multiple" and _num(e.get("delta")) is not None:
        ex = f"|차이|={fmt_value(e['delta'])}" + (f", {ex}" if ex else "")
    return _clip(f"{base} ({ex})" if base and ex else base or ex, 2 * MAX_TEXT)


def _compare_tol(section: str, kind: str, e: dict) -> str:
    thr = _num(e.get("threshold"))
    if section == "게이트":
        if kind == "exact":
            return "정확 일치"
        if kind == "sign":
            return "양(+) 차이 시드 수 같음"
        if thr is not None:
            return f"|평균 차이| ≤ {fmt_value(thr)}" if kind == "mean" else f"|차이| ≤ {fmt_value(thr)}"
        return ""
    if section == "보고":
        if kind == "config":
            return "(보고) 정확 일치"
        if thr is not None:
            return f"(보고 기준) |차이| ≤ {fmt_value(thr)}"
    return ""


def _compare_row(section: str, e: dict, runvals: dict) -> dict:
    kind = str(e.get("kind") or e.get("gate") or "")
    metric = str(e.get("metric") or "")
    ref, new = e.get("ref", _MISSING), e.get("new", _MISSING)
    if kind == "sd_multiple" and ref is _MISSING and new is _MISSING:   # 같은 런·지표의 single_run 값을 빌린다
        ref, new = runvals.get((str(e.get("exp")), metric), (_MISSING, _MISSING))
    if section == "게이트":
        verdict = "PASS" if e.get("status") == "PASS" else "FAIL"          # 보고서의 gate 검사와 같은 규칙
    elif section == "미평가":
        verdict = "N/A"
    else:
        verdict = to_verdict(e.get("status"))
    return {"구분": section, "종류": kind, "실험": _who(e), "지표": metric, COMMITTED: fmt_value(ref),
            NEW: fmt_value(new), DIFF: fmt_diff(signed_diff(ref, new)), TOL: _compare_tol(section, kind, e),
            VERDICT: verdict, NOTE: _note_of(e)}


def _gate_row(g: dict) -> dict:
    kind = str(g.get("kind") or "")
    name = f"{kind}: {_who(g)}/{g.get('metric') or ''}".rstrip("/")
    delta, thr = _num(g.get("delta")), _num(g.get("threshold"))
    if kind == "exact":
        stat, val, tol = "값 일치", fmt_value(g.get("new", _MISSING)), "= " + fmt_value(g.get("ref", _MISSING))
    elif kind == "sign":
        stat, val, tol = "양(+) 차이 시드 수", fmt_value(g.get("new", _MISSING)), "= " + fmt_value(g.get("ref", _MISSING))
    else:
        stat = ("|새 평균 − 커밋 평균|" + (f" ({fmt_value(g['n'])}시드)" if g.get("n") is not None else "")
                if kind == "mean" else "|새 값 − 커밋 값|") if delta is not None else ""
        val = fmt_diff(delta)
        tol = f"≤ {fmt_value(thr)}" if thr is not None else ""
    return {"게이트": name, "통계량": stat, "값": val, TOL: tol,
            VERDICT: "PASS" if g.get("status") == "PASS" else "FAIL", NOTE: _note_of(g)}


def compare_tables(dest, res) -> list[Table]:
    """compare → compare_table·gates. res = nmfc_compare.retrain_compare 결과(compare_result.json과 같은 내용).

    compare_table: 전제조건 불충족 사유(FAIL), 게이트(PASS·FAIL), 미평가(N/A), 보고(file_change 제외, REPORT·N/A).
    gates: 전제조건 불충족 사유(FAIL), 게이트, 미평가(N/A). file_change 보고는 파일 단위라 표에 넣지 않고 건수만 적는다."""
    res = res if isinstance(res, dict) else {}
    gates = [g for g in res.get("gates") or [] if isinstance(g, dict)]
    reports = [r for r in res.get("reports") or [] if isinstance(r, dict)]
    uneval = [u for u in res.get("unevaluated") or [] if isinstance(u, dict)]
    pre = res.get("precondition") if isinstance(res.get("precondition"), dict) else {}
    reasons = [] if pre.get("ok") else [str(x) for x in pre.get("reasons") or []]
    runvals = {(str(e.get("exp")), str(e.get("metric"))): (e.get("ref", _MISSING), e.get("new", _MISSING))
               for e in gates + reports if e.get("kind") == "single_run" and e.get("exp")}
    pre_rows = [{"구분": "전제조건", VERDICT: "FAIL", NOTE: r} for r in reasons]
    ct = list(pre_rows)
    ct += [_compare_row("게이트", g, runvals) for g in gates]
    ct += [_compare_row("미평가", u, runvals) for u in uneval]
    ct += [_compare_row("보고", r, runvals) for r in reports if r.get("kind") != "file_change"]
    gt = [{"게이트": "전제조건", VERDICT: "FAIL", NOTE: r} for r in reasons]
    gt += [_gate_row(g) for g in gates]
    gt += [{"게이트": f"{u.get('gate') or '미평가'}: {_who(u)}/{u.get('metric') or ''}".rstrip("/"),
            VERDICT: "N/A", NOTE: _note_of(u)} for u in uneval]
    summ = res.get("summary") if isinstance(res.get("summary"), dict) else {}
    notes = [f"compare 종료코드 {res.get('exit_code')}: {summ.get('reason') or '-'}",
             ("전제조건: 충족" if pre.get("ok") else
              "전제조건: 불충족 — " + ("; ".join(reasons) if reasons else "결과 없음")),
             f"학습한 런 {len(res.get('trained') or [])}개 · 판정에 쓴 실행 {len(res.get('runs') or [])}개"]
    fc = [r for r in reports if r.get("kind") == "file_change"]
    fc_warn = sum(1 for r in fc if r.get("status") == "WARN")
    ct_notes = notes + ([f"파일 변화 보고(kind file_change) {len(fc)}건(그중 예상 밖 변경·없어진 파일 = compare 보고서의 "
                         f"WARN {fc_warn}건)은 이 표에 넣지 않았다 — compare_result.json의 reports와 compare 보고서에 있다"]
                        if fc else [])
    return [write_table(dest, "compare_table", "compare 결과표 — compare_table (실험 × 지표)", COMPARE_COLUMNS, ct,
                        ct_notes),
            write_table(dest, "gates", "compare 결과표 — gates", GATE_COLUMNS, gt, notes)]
