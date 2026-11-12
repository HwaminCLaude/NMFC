"""nmfc_compare — NMFC 재현 판정기(nmfc verify·regen·compare가 쓰는 비교 함수).

독립 모듈이다. 다른 nmfc_ 모듈을 import하지 않는다. LF 정규화와 이진 판별은 nmfc_common과 같은
규칙(BINARY_EXT, 앞 8000바이트 NUL 휴리스틱, CRLF→LF)으로 여기에 따로 둔다. 최상위 import는
stdlib·yaml·numpy뿐이고 PIL·scipy는 함수 안에서 지연 import한다.

명령별 진입점
  verify   compare_written · png_compare · compare_float_platform
  regen    compare_json_rules · regen_compare · regen_label (· regen_mode)
  compare  t_factor · retrain_compare

finding의 status 어휘는 Report와 같다: PASS · WARN · REPORT · FAIL · N/A.
regen에서 값이 완전히 같은 리프·CSV 셀·파일은 finding을 만들지 않는다(개수는 counts에 센다).
같은 항목까지 행으로 받으려면 regen_compare(..., include_identical=True)의 "rows"를 쓴다(행 형식은 그 문서열).

이 모듈이 정한 세부
 1. JSON 경로는 '/'로 잇는다. 패턴의 '*'는 경로 성분 하나 안에서만, '**'는 성분 0개 이상과 맞는다
    (fnmatchcase, 대소문자 구분 — Windows와 리눅스에서 결과가 같다). stats_ci.json의 positive_episodes·
    n_episodes·exceeds_band는 깊이 3(fewshot_k1_paired_delta/<exp>/…)이라 tolerances.yaml에는 `**/…`로 적었다.
 2. 리스트의 원소가 모두 dict이고 행 키(rows_key, 기본 'exp') 값이 양쪽에서 유일하면 원소를 그 값으로 짝짓는다
    (retrieval_eval.json의 행, openset_noise_grid/master.json의 rows). rows_key가 목록이면 그 필드 값들을 ':'로
    이은 복합 키로 짝짓는다(calibration_eval.json의 rows는 [exp, variant] → 경로 성분 예: c100v_shrb:raw).
    그 밖의 리스트는 인덱스로 짝짓는다. 한쪽에만 있는 행은 경로 누락·추가(FAIL)다. 기준본은 행 키로 짝지을 수
    있는데 새 파일은 못 하면(중복·누락) FAIL이다. 공통 행의 순서가 바뀌면 WARN이다.
 3. 형 우선 규칙: bool·int·str·null·형 불일치는 규칙과 무관하게 정확 일치를 요구한다.
    허용오차·보고 규칙은 양쪽이 모두 float인 리프에만 적용한다.
 4. 허용오차 비교는 |Δ| ≤ tol·(1+1e-9)로 한다(허용오차 경계값에서 부동소수 잡음으로 떨어지지 않게).
 5. compare_json_rules·regen_compare 결과의 'gate'가 False인 finding(보고 전용: purity·nmi, 선택 파일 N/A,
    font_dependent PNG)만 PASS_EXACT를 막지 않는다. 그 밖의 비동일 finding(PASS·WARN 포함)은 PASS_EXACT를 막는다.
    {report: true, blocks_exact: true} 규칙(ckpt 평가기 산출물)과 files의 on_diff: report는 REPORT(FAIL 아님)이지만
    gate가 True라 PASS_EXACT를 막는다(PASS_EXACT는 purity·nmi를 뺀 모든 리프가 같아야 한다).
    files의 optional: true는 한쪽에 파일이 없으면 FAIL 대신 N/A(gate False)다.
 6. compare_float_platform: 숫자 리프(int·float)는 math.isclose(rel, abs). 문자열 리프는 NUM_RE로
    숫자를 뽑아 같은 규칙으로 비교하고 나머지 텍스트는 정확히 비교한다(re.ASCII). 정확 일치 경로는 키워드
    인자 exact_paths(기본 checks/*/name, checks/*/passed, n_passed, n_total)다. 리스트는 인덱스로 짝짓는다.
 7. regen_compare는 기준본이 없는 비선택 파일을 FAIL로 본다(공허한 PASS_EXACT 방지). optional 파일은 N/A.
 8. retrain_compare는 키 gates·reports·unevaluated·pass에 더해 exit_code(0·1·2),
    precondition{ok, reasons}, trained, summary, runs(실행별 요약)를 돌려준다. CLI는 exit_code를 그대로 종료코드로 쓴다.
    - 전제조건: 학습한 실험 ≥1, 각 metrics.json mtime ≥ 실행 시작(그 실험을 학습한 items[*]의
      started_ns·started가 있으면 그것, 없으면 최상위 started_ns·started; 1초 여유), 컨테이너에서 results/가 마운트.
      하나라도 아니면 exit_code 2.
    - 실행 단위(status['runs'] = nmfc_cli.merged_status가 붙인 실행별 상태, 없으면 status 하나): 마지막 실행은
      pass가 True이고 exit_code(있으면)가 0이어야 한다(실패·중단·진행 중이면 2). 그 앞의 실패 실행은 실패·중단
      항목이 뒤의 ok 항목으로 해소되어야 한다(학습: 끝내지 못한 런이 뒤의 ok 학습 항목 trained에 있음, 단계: 같은 id가
      뒤에서 ok). 마지막 실행이 학습 없이 성공한 경우(예: 학습 뒤 reproduce --from S3)는 막지 않는다.
    - 학습 산출물 지문(items[*].artifacts, nmfc_pipeline이 학습 직후 기록): 있으면 지금의 metrics.json·embeddings.npz가
      기록한 원시 sha256·size와 같아야 하고(되돌리기·덮어쓰기 탐지), npz는 그 학습에서 다시 쓰였어야 한다
      (rewritten). 지문과 같은데 기준본과 내용까지 같으면 WARN(kind identical)으로 보고만 한다. 지문이 없으면(이전 형식)
      metrics.json이 기준본과 LF 정규화 내용이 같거나 embeddings.npz가 실행 시작보다 오래됐으면 전제조건 불충족이다.
    - pass = 전제조건 충족 ∧ 평가된 게이트 ≥1 ∧ FAIL 게이트 0 ∧ 미평가 0(fail_on_unevaluated). 미평가만 있으면 2.
    - 집계 출처(seed_robustness.json·near_ood.json)는 런의 embeddings.npz(없으면 metrics.json)보다 오래되었거나
      기준본과 내용이 같으면 미평가다. metrics.json 출처 지표는 런 단위 전제조건으로 신선도를 본다.
    - SD는 기준본의 시드 값들로 계산한 표본 SD다(core·ood는 seed_robustness.json의 sd와 같은 값, grid는
      master.json 셀별 4시드 SD와 같은 값).
    - 단일 런 게이트는 해당 역할(셀)의 시드가 전부 재학습되지 않았을 때 쓰고, 전부 재학습되면 5시드(그리드 4시드)
      평균 게이트를 쓰며 런별 결과는 보고로 내린다. m_run은 가족의 게이트 지표 수(core 5, ood 2, grid 2)다.
    - transition_epoch: 기준본 variant.apt가 reciprocal이면 정확 일치, variant_code가 report_variant_codes
      (tolerances.yaml: shjb·shjo, negJ 계열)에 있으면 보고만. apt가 없는 런(softmax·합성·cifar100_nmfc 등)은 보고만.
    - config_keys(runner·dataset·method·epochs 등) 불일치는 WARN 보고다(게이트 아님).
"""

from __future__ import annotations

import collections
import csv
import fnmatch
import hashlib
import io
import json
import math
import os
import re
import statistics
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

# ============================================================================ 공통

_MISSING = object()

BINARY_EXT = frozenset({".png", ".jpg", ".jpeg", ".npz", ".npy", ".pt", ".pth", ".pdf",
                        ".pptx", ".zip", ".xlsx", ".mat", ".gz"})
ROWS_KEY = "exp"
ROW_KEY_SEP = ":"   # 복합 행 키(rows_key가 목록)의 값을 이어 경로 성분 하나로 만든다(예: c100v_shrb:raw)
NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", re.ASCII)        # float_platform 문자열 속 숫자
FP_EXACT_PATHS = ("checks/*/name", "checks/*/passed", "n_passed", "n_total")
_SLACK = 1e-9


def _is_binary(path, data: bytes) -> bool:
    return Path(str(path)).suffix.lower() in BINARY_EXT or b"\x00" in data[:8000]


def _norm_bytes(path, data: bytes) -> bytes:
    return data if _is_binary(path, data) else data.replace(b"\r\n", b"\n")


def _norm_sha256(path) -> str:
    p = Path(path)
    return hashlib.sha256(_norm_bytes(p, p.read_bytes())).hexdigest()


# PyYAML(YAML 1.1)은 '1e-5'·'1.0e5'를 문자열로 읽는다. 지수 부호가 있거나 가수에 점이 있는 지수 표기만 float로 바꾼다
# (점·부호가 없는 '6e1234' 같은 해시 조각은 그대로 둔다).
_SCI_RE = re.compile(r"[-+]?(?:(?:\d+\.\d*|\.\d+|\d+)[eE][-+]\d+|(?:\d+\.\d*|\.\d+)[eE]\d+)")


def _coerce_sci(obj):
    if isinstance(obj, dict):
        return {k: _coerce_sci(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_coerce_sci(v) for v in obj]
    if isinstance(obj, str) and _SCI_RE.fullmatch(obj.strip()):
        return float(obj)
    return obj


def load_yaml(path: Path) -> dict:
    """UTF-8 YAML을 읽는다. 지수 표기 문자열은 float로 바꾼다. 빈 파일이면 {}."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return _coerce_sci(data) if data is not None else {}


def _load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _load_json_cached(path: Path, cache: dict):
    key = str(path)
    if key not in cache:
        try:
            cache[key] = _load_json(path)
        except (OSError, ValueError):
            cache[key] = _MISSING
    return cache[key]


def _split(path) -> tuple:
    return tuple(p for p in str(path).replace("\\", "/").split("/") if p != "")


def _glob_match(pattern, parts) -> bool:
    """'*'는 성분 하나 안, '**'는 성분 0개 이상. 성분 비교는 fnmatchcase."""
    pat = _split(pattern) if isinstance(pattern, str) else tuple(pattern)
    parts = tuple(parts)
    memo: dict = {}

    def rec(i, j):
        if (i, j) in memo:
            return memo[(i, j)]
        if i == len(pat):
            r = j == len(parts)
        elif pat[i] == "**":
            r = any(rec(i + 1, k) for k in range(j, len(parts) + 1))
        else:
            r = j < len(parts) and fnmatch.fnmatchcase(parts[j], pat[i]) and rec(i + 1, j + 1)
        memo[(i, j)] = r
        return r

    return rec(0, 0)


def _is_float(x) -> bool:
    return isinstance(x, float)


def _num(x):
    """int·float(bool 제외)면 float, 아니면 None."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return float(x)


def _same(a, b) -> bool:
    """형까지 같은 동일성(bool≠int, int≠float). NaN끼리는 같다고 본다."""
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _within(d: float, tol: float, a: float = 0.0, b: float = 0.0, rel: float = 0.0) -> bool:
    if d is None or math.isnan(d):
        return False
    lim = max(tol or 0.0, (rel or 0.0) * max(abs(a), abs(b)))
    return d <= lim * (1.0 + _SLACK)


def _brief(v):
    """보고서용 값 요약: 스칼라는 그대로, 큰 컨테이너는 앞부분 문자열."""
    if isinstance(v, (dict, list)):
        s = json.dumps(v, ensure_ascii=False)
        return v if len(s) <= 200 else s[:200] + "…"
    if v is _MISSING:
        return None
    return v


def _row_key_fields(key) -> tuple:
    """rows_key(필드 이름 하나 또는 목록)를 필드 이름 튜플로. 없으면 ()."""
    if not key:
        return ()
    if isinstance(key, (list, tuple)):
        return tuple(str(k) for k in key)
    return (str(key),)


def _key_label(key) -> str:
    """보고 문구용 행 키 표기: 'exp' 또는 'exp+variant'."""
    return "+".join(_row_key_fields(key))


def _row_id(row: dict, fields: tuple) -> str:
    """행 키 값. 필드가 여럿이면 ROW_KEY_SEP로 잇는다. 짝짓기·경로 성분·_resolve가 모두 이 값을 쓴다."""
    return ROW_KEY_SEP.join(str(row[f]) for f in fields)


def _keyable(lst, key) -> bool:
    fields = _row_key_fields(key)
    if not fields or not isinstance(lst, list) or not lst:
        return False
    vals = []
    for r in lst:
        if not isinstance(r, dict) or any(f not in r for f in fields):
            return False
        for f in fields:
            v = r[f]
            if isinstance(v, bool) or not isinstance(v, (str, int)):
                return False
        vals.append(_row_id(r, fields))
    return len(set(vals)) == len(vals)


def _pair_events(ref, new, parts, rows_key):
    """두 JSON 트리를 함께 걷는다. (kind, parts, a, b), kind ∈ leaf·missing·extra·order·rowkey.
    rows_key는 필드 이름 하나 또는 목록(복합 키)이고 모든 깊이의 리스트에 같은 키를 시도한다."""
    if isinstance(ref, dict) and isinstance(new, dict):
        if not ref and not new:
            yield ("leaf", parts, ref, new)
            return
        for k, v in ref.items():
            kp = parts + (str(k),)
            if k in new:
                yield from _pair_events(v, new[k], kp, rows_key)
            else:
                yield ("missing", kp, v, _MISSING)
        for k, v in new.items():
            if k not in ref:
                yield ("extra", parts + (str(k),), _MISSING, v)
        return
    if isinstance(ref, list) and isinstance(new, list):
        if not ref and not new:
            yield ("leaf", parts, ref, new)
            return
        kr, kn = _keyable(ref, rows_key), _keyable(new, rows_key)
        if kr and kn:
            fields = _row_key_fields(rows_key)
            rmap = {_row_id(r, fields): r for r in ref}
            nmap = {_row_id(r, fields): r for r in new}
            for k, v in rmap.items():
                if k in nmap:
                    yield from _pair_events(v, nmap[k], parts + (k,), rows_key)
                else:
                    yield ("missing", parts + (k,), v, _MISSING)
            for k, v in nmap.items():
                if k not in rmap:
                    yield ("extra", parts + (k,), _MISSING, v)
            ro = [k for k in rmap if k in nmap]
            no = [k for k in nmap if k in rmap]
            if ro != no:
                yield ("order", parts, ro, no)
            return
        if kr and not kn:
            yield ("rowkey", parts, rows_key, None)
        for i in range(min(len(ref), len(new))):
            yield from _pair_events(ref[i], new[i], parts + (str(i),), rows_key)
        for i in range(len(new), len(ref)):
            yield ("missing", parts + (str(i),), ref[i], _MISSING)
        for i in range(len(ref), len(new)):
            yield ("extra", parts + (str(i),), _MISSING, new[i])
        return
    yield ("leaf", parts, ref, new)


def _resolve(obj, parts, rows_key=ROWS_KEY):
    """경로 성분을 따라 값을 찾는다. 리스트는 rows_key로 짝지을 수 있으면 키로, 아니면 인덱스로."""
    cur = obj
    for p in parts:
        if cur is _MISSING:
            return _MISSING
        if isinstance(cur, dict):
            cur = cur.get(p, _MISSING)
        elif isinstance(cur, list):
            if _keyable(cur, rows_key):
                fields = _row_key_fields(rows_key)
                hit = [r for r in cur if _row_id(r, fields) == p]
                cur = hit[0] if hit else _MISSING
            elif p.isdigit() and int(p) < len(cur):
                cur = cur[int(p)]
            else:
                return _MISSING
        else:
            return _MISSING
    return cur


def _dotted(d, key):
    cur = d
    for p in str(key).split("."):
        if not isinstance(cur, dict) or p not in cur:
            return _MISSING
        cur = cur[p]
    return cur


# ============================================================================ verify

def png_compare(a: Path, b: Path) -> dict:
    """Pillow로 디코딩한 RGBA 배열 비교. PNG 압축 바이트 차이는 무시한다."""
    from PIL import Image

    with Image.open(a) as ia:
        xa = np.asarray(ia.convert("RGBA"))
    with Image.open(b) as ib:
        xb = np.asarray(ib.convert("RGBA"))
    out = {"size_equal": xa.shape == xb.shape, "pixel_equal": False, "changed_ratio": None,
           "max_channel_diff": None, "size_a": [int(xa.shape[1]), int(xa.shape[0])],
           "size_b": [int(xb.shape[1]), int(xb.shape[0])]}
    if xa.shape != xb.shape:
        return out
    diff = np.abs(xa.astype(np.int16) - xb.astype(np.int16))
    changed = diff.any(axis=-1)
    out["pixel_equal"] = not bool(changed.any())
    out["changed_ratio"] = float(changed.mean()) if changed.size else 0.0
    out["max_channel_diff"] = int(diff.max()) if diff.size else 0
    return out


def _compare_numeric_strings(a: str, b: str, rel: float, abs_: float):
    """(ok, detail, data). 숫자를 뺀 텍스트는 정확히, 숫자는 isclose로."""
    na, nb = NUM_RE.findall(a), NUM_RE.findall(b)
    sa, sb = NUM_RE.sub("\x00", a), NUM_RE.sub("\x00", b)
    data = {"numbers_ref": na, "numbers_new": nb}
    if sa != sb or len(na) != len(nb):
        return False, "숫자를 뺀 텍스트가 다르다", data
    worst = 0.0
    for x, y in zip(na, nb):
        fx, fy = float(x), float(y)
        if not math.isclose(fx, fy, rel_tol=rel, abs_tol=abs_):
            return False, f"문자열 속 숫자 {x} → {y}가 허용오차 밖", data
        worst = max(worst, abs(fy - fx))
    data["max_abs_delta"] = worst
    return True, f"문자열 속 숫자만 다름(max|Δ|={worst:.3g})", data


def compare_float_platform(ref, new, rel=1e-9, abs_=1e-12, *, exact_paths=FP_EXACT_PATHS) -> list[dict]:
    """float_platform 판정. 비동일 리프마다 finding(PASS=허용오차 안, FAIL) 하나. 빈 목록이면 완전 동일."""
    exact_pats = [_split(p) for p in exact_paths]
    out = []
    for kind, parts, a, b in _pair_events(ref, new, (), None):
        path = "/".join(parts)
        if kind in ("missing", "extra"):
            out.append({"path": path, "status": "FAIL", "ref": _brief(a), "new": _brief(b),
                        "detail": "키 누락" if kind == "missing" else "키 추가"})
            continue
        if kind != "leaf" or _same(a, b):
            continue
        f = {"path": path, "status": "FAIL", "ref": _brief(a), "new": _brief(b), "detail": ""}
        if any(_glob_match(p, parts) for p in exact_pats):
            f["detail"] = "정확 일치 항목"
        elif _num(a) is not None and _num(b) is not None:
            d = abs(float(b) - float(a))
            f["delta"] = d
            if math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_):
                f["status"], f["detail"] = "PASS", f"|Δ|={d:.3g} ≤ isclose(rel={rel:g}, abs={abs_:g})"
            else:
                f["detail"] = f"|Δ|={d:.3g} > isclose(rel={rel:g}, abs={abs_:g})"
        elif isinstance(a, str) and isinstance(b, str):
            ok, detail, data = _compare_numeric_strings(a, b, rel, abs_)
            f["status"], f["detail"], f["data"] = ("PASS" if ok else "FAIL"), detail, data
        else:
            f["detail"] = "형·값 불일치"
        out.append(f)
    return out


def _snapshot_entry(before: dict, rel: str):
    if not before:
        return None
    for k in (rel, rel.replace("/", "\\"), rel.replace("\\", "/")):
        if k in before:
            return before[k]
    return None


def compare_written(baseline: Path, current: Path, writes: list[dict], before: dict) -> list[dict]:
    """verify: 다시 쓴 파일의 신선도(mtime_ns 변화)와 class별 내용 판정. 파일마다 finding 하나.

    finding = {"path", "class", "status", "detail", "data"}.
    font_dependent는 신선하면 REPORT(픽셀 동일 여부와 차이 비율 기록), 신선하지 않으면 FAIL이다.
    """
    baseline, current = Path(baseline), Path(current)
    out = []
    for w in writes or []:
        rel = str(w["path"]).replace("\\", "/")
        cls = w.get("class", "exact")
        cur, base = current / rel, baseline / rel
        snap = _snapshot_entry(before, rel)
        data: dict = {"fresh": False}
        problems: list[str] = []
        if not cur.exists():
            problems.append("실행 뒤 파일이 없음")
        else:
            m_after = cur.stat().st_mtime_ns
            data["mtime_ns_after"] = m_after
            if snap is None:
                problems.append("사전 스냅샷에 항목이 없어 신선도를 판정할 수 없음")
            elif not snap.get("exists", True):
                data["fresh"] = True
            else:
                data["mtime_ns_before"] = snap.get("mtime_ns")
                data["fresh"] = snap.get("mtime_ns") is not None and m_after != snap.get("mtime_ns")
                if not data["fresh"]:
                    problems.append("다시 쓰이지 않음(mtime_ns 불변)")
        content_ok, content_detail, report_only = None, "", cls == "font_dependent"
        if cur.exists():
            if not base.exists():
                content_ok, content_detail = False, "기준본 없음"
            else:
                try:
                    if cls == "exact":
                        ha, hb = _norm_sha256(base), _norm_sha256(cur)
                        data.update(sha256_baseline=ha, sha256_current=hb)
                        content_ok = ha == hb
                        content_detail = "LF 정규화 sha256 일치" if content_ok else "LF 정규화 sha256 불일치"
                    elif cls in ("pixel_equal", "font_dependent"):
                        r = png_compare(base, cur)
                        data.update(r)
                        content_ok = r["pixel_equal"]
                        content_detail = ("RGBA 픽셀 동일" if content_ok else
                                          f"픽셀 차이(크기 동일={r['size_equal']}, 비율={r['changed_ratio']},"
                                          f" 최대 채널차={r['max_channel_diff']})")
                    elif cls == "float_platform":
                        fp = compare_float_platform(_load_json(base), _load_json(cur))
                        bad = [f for f in fp if f["status"] == "FAIL"]
                        data["findings"] = fp
                        content_ok = not bad
                        content_detail = (f"float_platform 통과(비동일 리프 {len(fp)}개 모두 허용오차 안)"
                                          if content_ok else f"float_platform 실패 {len(bad)}건: "
                                          + "; ".join(f"{f['path']}: {f['detail']}" for f in bad[:5]))
                    else:
                        content_ok, content_detail, report_only = False, f"알 수 없는 class: {cls}", False
                except Exception as e:  # 디코딩·파싱 실패는 판정 실패다
                    content_ok, content_detail, report_only = False, f"비교 중 오류: {e!r}", False
        if problems:
            status = "FAIL"
        elif report_only:
            status = "REPORT"
        else:
            status = "PASS" if content_ok else "FAIL"
        detail = "; ".join(problems + ([content_detail] if content_detail else []))
        out.append({"path": rel, "class": cls, "status": status, "detail": detail, "data": data})
    return out


# ============================================================================ regen — JSON 규칙

def _parse_check(spec) -> dict:
    if spec is None or spec == "exact":
        return {"kind": "exact", "tol": 0.0, "rel": 0.0, "on_diff": "fail", "desc": "exact"}
    if spec == "report":
        return {"kind": "report", "tol": None, "rel": 0.0, "on_diff": "report", "desc": "report"}
    if isinstance(spec, dict):
        tol = None
        if "abs" in spec:
            tol = float(spec["abs"])
        if "quantum" in spec:
            q = spec["quantum"]
            tol = float(spec.get("quanta", 1)) / (float(q["episodes"]) * float(q["queries"]))
        rel = float(spec.get("rel", 0.0))
        on_diff = str(spec.get("on_diff", "pass"))
        if on_diff not in ("pass", "warn"):
            raise ValueError(f"on_diff는 pass|warn: {spec!r}")
        if spec.get("report"):
            blocks = bool(spec.get("blocks_exact", False))
            desc = "report" + (f" (abs {tol:g})" if tol is not None else "") + (", PASS_EXACT 차단" if blocks else "")
            return {"kind": "report", "tol": tol, "rel": rel, "on_diff": "report", "desc": desc,
                    "blocks_exact": blocks}
        if tol is None and rel == 0.0:
            raise ValueError(f"허용오차 규칙에 abs·quantum·rel이 없음: {spec!r}")
        desc = f"abs {tol:g}" if tol is not None else f"rel {rel:g}"
        if "quantum" in spec:
            desc = f"quantum×{spec.get('quanta', 1)} (={tol:.6g})"
        if on_diff == "warn":
            desc += ", WARN"
        return {"kind": "tol", "tol": tol or 0.0, "rel": rel, "on_diff": on_diff, "desc": desc}
    raise ValueError(f"알 수 없는 규칙 형식: {spec!r}")


def _compile_rules(rules) -> list[dict]:
    out = []
    for r in rules or []:
        if "all" in r:
            ref_c = other_c = r["all"]
        else:
            ref_c, other_c = r.get("ref", "exact"), r.get("other", "exact")
        out.append({"pattern": r["path"], "parts": _split(r["path"]),
                    "ref": _parse_check(ref_c), "other": _parse_check(other_c)})
    return out


def _rule_for(compiled, parts, mode):
    for r in compiled:
        if _glob_match(r["parts"], parts):
            return r["pattern"], r[mode]
    return "(규칙 없음)", _parse_check("exact")


def _judge_leaf(name, parts, a, b, pattern, check) -> dict:
    f = {"file": name, "path": "/".join(parts), "rule": pattern, "check": check["desc"],
         "ref": _brief(a), "new": _brief(b), "delta": None, "tol": check.get("tol"),
         "gate": True, "status": "FAIL", "detail": ""}
    if not (_is_float(a) and _is_float(b)):
        f["detail"] = "bool·int·str·null·구조는 정확 일치여야 함(형 또는 값이 다름)"
        return f
    d = abs(b - a) if not (math.isnan(a) or math.isnan(b)) else float("nan")
    f["delta"] = d
    if check["kind"] == "exact":
        f["detail"] = f"정확 일치 필요(|Δ|={d:.3g})"
    elif check["kind"] == "report":
        # 보고만 한다(FAIL 아님). blocks_exact면 값이 다를 때 PASS_EXACT를 막는다(gate True → PASS_TOLERANCE).
        f["status"], f["gate"] = "REPORT", bool(check.get("blocks_exact", False))
        if check.get("tol") is not None:
            within = _within(d, check["tol"], a, b, check.get("rel", 0.0))
            f["data"] = {"within_tol": within}
            f["detail"] = f"보고만(|Δ|={d:.3g}, abs {check['tol']:g} {'이내' if within else '초과'})"
        else:
            f["detail"] = f"보고만(|Δ|={d:.3g})"
        if f["gate"]:
            f["detail"] += " — 값이 달라 PASS_EXACT는 아님"
    else:
        if _within(d, check["tol"], a, b, check.get("rel", 0.0)):
            f["status"] = "WARN" if check["on_diff"] == "warn" else "PASS"
            f["detail"] = f"허용오차 안(|Δ|={d:.3g} ≤ {check['desc']})"
        else:
            f["detail"] = f"허용오차 초과(|Δ|={d:.3g} > {check['desc']})"
    return f


def _signed_diff(a, b):
    """표시용 부호 있는 차이 new−ref. 양쪽이 수(bool 제외)이고 결과가 NaN이 아닐 때만, 아니면 None."""
    x, y = _num(a), _num(b)
    if x is None or y is None:
        return None
    d = y - x
    return None if math.isnan(d) else d


def _row(f: dict, section: str, kind: str, tol_text, *, identical: bool = False) -> dict:
    """finding(또는 같은 키를 가진 동일 항목 dict)을 표 행으로 복사한다. f 자체는 바꾸지 않는다."""
    return {**f, "section": section, "kind": kind, "tol_text": tol_text,
            "diff": _signed_diff(f.get("ref"), f.get("new")), "identical": identical}


def _add_row(rows, f: dict, section: str, kind: str, tol_text, *, identical: bool = False) -> None:
    if rows is not None:
        rows.append(_row(f, section, kind, tol_text, identical=identical))


def _identical_leaf_row(name, parts, a, b, pattern, check) -> dict:
    """값이 같은 JSON 리프의 표 행. 판정 규칙(exact·허용오차)이면 PASS, 보고 규칙이면 REPORT이고 gate는 False."""
    report = check["kind"] == "report"
    f = {"file": name, "path": "/".join(parts), "rule": pattern, "check": check["desc"],
         "ref": _brief(a), "new": _brief(b), "delta": 0.0 if (_is_float(a) and not math.isnan(a)) else None,
         "tol": check.get("tol"), "gate": False, "status": "REPORT" if report else "PASS",
         "detail": "값 동일(보고 규칙)" if report else "값 동일"}
    return _row(f, "json", "leaf", check["desc"], identical=True)


def _compare_json_impl(name, ref, new, mode, rules, rows_key=ROWS_KEY, rows=None):
    """(findings, counts). rows가 목록이면 비교한 모든 리프(값이 같은 리프 포함)와 구조 finding을
    표 행(regen_compare 문서열의 형식)으로 덧붙인다. findings·counts는 rows 유무와 무관하게 같다."""
    if mode not in ("ref", "other"):
        raise ValueError(f"mode는 ref|other: {mode!r}")
    compiled = _compile_rules(rules)
    findings, counts = [], {"leaves": 0, "identical": 0}
    for kind, parts, a, b in _pair_events(ref, new, (), rows_key):
        path = "/".join(parts)
        if kind == "leaf":
            counts["leaves"] += 1
            if _same(a, b):
                counts["identical"] += 1
                if rows is not None:
                    pattern, check = _rule_for(compiled, parts, mode)
                    rows.append(_identical_leaf_row(name, parts, a, b, pattern, check))
                continue
            pattern, check = _rule_for(compiled, parts, mode)
            f = _judge_leaf(name, parts, a, b, pattern, check)
            row_kind, tol_text = "leaf", check["desc"]
        elif kind in ("missing", "extra"):
            f = {"file": name, "path": path, "rule": "(경로 집합)", "check": "structure",
                 "ref": _brief(a), "new": _brief(b), "delta": None, "tol": None, "gate": True,
                 "status": "FAIL",
                 "detail": "새 파일에 경로 없음" if kind == "missing" else "기준본에 없는 경로"}
            row_kind, tol_text = "structure", "경로 집합 동일"
        elif kind == "order":
            f = {"file": name, "path": path or "(최상위)", "rule": "(행 순서)", "check": "order",
                 "ref": a, "new": b, "delta": None, "tol": None, "gate": True, "status": "WARN",
                 "detail": f"'{_key_label(rows_key)}' 행 순서가 다름"}
            row_kind, tol_text = "order", "행 순서 동일"
        elif kind == "rowkey":
            f = {"file": name, "path": path or "(최상위)", "rule": "(행 키)", "check": "rows_key",
                 "ref": None, "new": None, "delta": None, "tol": None, "gate": True,
                 "status": "FAIL",
                 "detail": f"기준본은 '{_key_label(a)}'로 행을 짝지을 수 있으나 새 파일은 '{_key_label(a)}'가 중복·누락됨"}
            row_kind, tol_text = "rows_key", "행 키 유일"
        else:
            continue
        findings.append(f)
        _add_row(rows, f, "json", row_kind, tol_text)
    return findings, counts


def compare_json_rules(name: str, ref, new, mode: str, rules: list[dict], *,
                       rows_key: str | list[str] = ROWS_KEY) -> list[dict]:
    """tolerances.yaml json 규칙 판정. 비동일 리프·구조 차이마다 finding 하나(동일 리프는 생략).

    finding = {"file","path","rule","check","ref","new","delta","tol","gate","status","detail"[, "data"]}.
    규칙은 위에서부터 처음 일치하는 것을 쓴다. 일치하는 규칙이 없으면 exact.
    rows_key는 행 키 필드 이름 하나 또는 목록(복합 키, 예: ["exp", "variant"]).
    """
    return _compare_json_impl(name, ref, new, mode, rules, rows_key)[0]


# ============================================================================ regen — 파일 묶음

def _read_csv_cells(path: Path) -> list[list[str]]:
    text = Path(path).read_bytes().decode("utf-8-sig")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return list(csv.reader(io.StringIO(text)))


def _compare_csv(rel, base: Path, cur: Path, limit: int = 50, *, rows=None):
    """CSV 셀 단위 정확 비교. (finding 목록, {"cells","identical"}). finding은 행·열 수 차이 포함 limit개까지다.
    rows가 목록이면 행·열 수 차이와 비교한 모든 셀(같은 셀 포함, 개수 상한 없음)을 표 행으로 덧붙인다."""
    a, b = _read_csv_cells(base), _read_csv_cells(cur)
    findings, cells, same = [], 0, 0
    out_rows: list[dict] = []
    if len(a) != len(b):
        f = {"file": rel, "path": "(행 수)", "rule": "csv_cells", "check": "csv_cells", "ref": len(a),
             "new": len(b), "delta": None, "tol": None, "gate": True, "status": "FAIL",
             "detail": "CSV 행 수가 다름"}
        findings.append(f)
        out_rows.append(_row(f, "files", "csv_shape", "행·열 수 동일"))
    for i, (ra, rb) in enumerate(zip(a, b)):
        if len(ra) != len(rb):
            f = {"file": rel, "path": f"r{i}", "rule": "csv_cells", "check": "csv_cells", "ref": len(ra),
                 "new": len(rb), "delta": None, "tol": None, "gate": True, "status": "FAIL",
                 "detail": "CSV 열 수가 다름"}
            findings.append(f)
            out_rows.append(_row(f, "files", "csv_shape", "행·열 수 동일"))
            continue
        for j, (x, y) in enumerate(zip(ra, rb)):
            cells += 1
            if x == y:
                same += 1
                if rows is not None:
                    out_rows.append(_row({"file": rel, "path": f"r{i}c{j}", "rule": "csv_cells", "check": "csv_cells",
                                          "ref": x, "new": y, "delta": None, "tol": None, "gate": False,
                                          "status": "PASS", "detail": "셀 동일"},
                                         "files", "csv_cell", "셀 정확 일치", identical=True))
                continue
            f = {"file": rel, "path": f"r{i}c{j}", "rule": "csv_cells", "check": "csv_cells",
                 "ref": x, "new": y, "delta": None, "tol": None, "gate": True, "status": "FAIL",
                 "detail": "셀 불일치(LF 정규화 후 셀 단위 정확 비교)"}
            if len(findings) < limit:
                findings.append(f)
            out_rows.append(_row(f, "files", "csv_cell", "셀 정확 일치"))
    if rows is not None:
        rows.extend(out_rows)
    return findings, {"cells": cells, "identical": same}


def _file_targets(baseline: Path, current: Path, spec: dict) -> list[str]:
    if "glob" in spec:
        found = set()
        for root in (baseline, current):
            for p in root.glob(spec["glob"]):
                if p.is_file():
                    found.add(p.relative_to(root).as_posix())
        return sorted(found)
    return [str(spec["path"]).replace("\\", "/")]


def regen_compare(baseline: Path, current: Path, mode: str, tol: dict, *, include_identical: bool = False) -> dict:
    """regen 판정: tolerances.yaml의 json·files 절을 baseline(기준본 루트)과 current(작업 루트)에 적용한다.

    반환 {"findings": [...], "counts": {...}}. include_identical=True면 "rows": [...]도 넣는다(표 작성용).
    findings·counts(따라서 regen_label 결과)는 include_identical과 무관하게 같다. regen_label은 rows를 보지 않는다.

    rows — 비교한 모든 항목을 한 행씩, tolerances.yaml 순서(json 절의 파일 순서, 그다음 files 절)로 담는다.
      json 절   파일마다 비교한 모든 JSON 리프(값이 같은 리프 포함, 트리 순회 순서)와 구조 차이.
                파일이 없거나 JSON으로 읽히지 않으면 그 파일은 한 행.
      files 절  csv_cells는 비교한 모든 셀(개수 상한 없음. findings의 셀 finding은 파일당 50개까지)과 행·열 수 차이.
                PNG(font_dependent·pixel_equal)와 exact 파일은 파일마다 한 행. 파일이 없으면 한 행.
    행 = finding의 키 전부(file, path, rule, check, ref, new, delta, tol, gate, status, detail[, data])를 복사하고
    아래 키를 더한 dict다(findings의 dict 자체는 바꾸지 않는다).
      section    "json" | "files"
      kind       json: "leaf" | "structure"(경로 누락·추가) | "order"(행 순서) | "rows_key" | "presence" | "error"
                 files: "csv_cell" | "csv_shape"(행·열 수) | "png" | "sha256" | "presence" | "error"
      tol_text   허용범위 표기. leaf는 규칙 설명(check와 같다. 예: "exact", "abs 1e-05", "abs 1e-06, WARN",
                 "quantum×1 (=5.05051e-06), WARN", "report", "report (abs 0.001)", "report, PASS_EXACT 차단"),
                 그 밖은 고정 문구: "경로 집합 동일", "행 순서 동일", "행 키 유일", "파일 있음",
                 "선택 파일(없으면 N/A)", "JSON 읽기", "셀 정확 일치", "행·열 수 동일", "RGBA 픽셀 동일",
                 "보고만(RGBA 픽셀 비교)", "LF 정규화 sha256 동일", "LF 정규화 sha256 동일(다르면 보고)".
      diff       부호 있는 차이 new−ref. 양쪽이 수(bool 제외)일 때만, 아니면 None.
      identical  값(파일 행은 내용)이 기준본과 같으면 True. 이때 gate는 False다.
    path    JSON은 '/'로 이은 경로(행 키로 짝지은 행은 키 값, 복합 키는 ':'로 이음. 예: c100v_shrb/mAP,
            rows/c100v_shrb:raw/ece). CSV 셀은 "r{i}c{j}"(0부터, r0 = 머리글), 행 수는 "(행 수)", 열 수는 "r{i}".
            파일 단위 행은 "".
    ref·new 기준본 값과 새 값(_brief: 직렬화가 200자를 넘는 컨테이너는 앞부분 문자열). CSV 셀은 문자열.
    delta   |new−ref|. 양쪽이 float일 때만(같은 float 리프는 0.0), 아니면 None.
    tol     규칙의 허용오차 수치(exact 0.0, abs·quantum 값, report는 abs 또는 None). 구조·파일 행은 None.
    data    PNG 행은 png_compare 결과(size_equal, pixel_equal, changed_ratio, max_channel_diff, size_a, size_b),
            report+abs 리프는 {"within_tol": bool}.
    status  다른 항목은 finding과 같다: FAIL, PASS(허용오차 안), WARN(허용오차 안이지만 on_diff warn, 또는 kind
            order), REPORT(보고 규칙·font_dependent·on_diff report), N/A(선택 파일 없음).
            같은 항목은 판정 규칙(exact·허용오차·csv_cells·pixel_equal·exact 파일)이면 PASS, 보고 규칙(report,
            blocks_exact, font_dependent, on_diff report)이면 REPORT다.
    """
    if mode not in ("ref", "other"):
        raise ValueError(f"mode는 ref|other: {mode!r}")
    baseline, current = Path(baseline), Path(current)
    findings: list[dict] = []  # json 절
    file_findings: list[dict] = []  # files 절
    rows: list[dict] | None = [] if include_identical else None
    jc = {"files": 0, "leaves": 0, "identical": 0}
    for rel, spec in (tol.get("json") or {}).items():
        spec = spec or {}
        optional = bool(spec.get("optional", False))
        bp, cp = baseline / rel, current / rel
        na = {"file": rel, "path": "", "rule": "(파일)", "check": "presence", "ref": None, "new": None,
              "delta": None, "tol": None}
        presence = "선택 파일(없으면 N/A)" if optional else "파일 있음"
        if not bp.exists():
            f = {**na, "gate": optional is False, "status": "N/A" if optional else "FAIL", "detail": "기준본에 파일 없음"}
            findings.append(f)
            _add_row(rows, f, "json", "presence", presence)
            continue
        if not cp.exists():
            f = {**na, "gate": optional is False, "status": "N/A" if optional else "FAIL",
                 "detail": "재생성 결과에 파일 없음"}
            findings.append(f)
            _add_row(rows, f, "json", "presence", presence)
            continue
        try:
            ref, new = _load_json(bp), _load_json(cp)
        except (OSError, ValueError) as e:
            f = {**na, "gate": True, "status": "FAIL", "detail": f"JSON 읽기 실패: {e!r}"}
            findings.append(f)
            _add_row(rows, f, "json", "error", "JSON 읽기")
            continue
        f, c = _compare_json_impl(rel, ref, new, mode, spec.get("rules", []), spec.get("rows_key", ROWS_KEY),
                                  rows=rows)
        findings += f
        jc["files"] += 1
        jc["leaves"] += c["leaves"]
        jc["identical"] += c["identical"]

    fc = {"files": 0, "cells": 0, "identical_cells": 0, "png": 0, "png_pixel_equal": 0}
    for spec in tol.get("files") or []:
        cls = spec.get("class", "exact")
        optional = bool(spec.get("optional", False))
        on_diff = str(spec.get("on_diff", "fail"))
        if on_diff not in ("fail", "report"):
            raise ValueError(f"files.on_diff는 fail|report: {spec!r}")
        tt = {"csv_cells": "셀 정확 일치", "pixel_equal": "RGBA 픽셀 동일", "font_dependent": "보고만(RGBA 픽셀 비교)",
              "exact": "LF 정규화 sha256 동일" + ("(다르면 보고)" if on_diff == "report" else "")}.get(cls, str(cls))
        for rel in _file_targets(baseline, current, spec):
            bp, cp = baseline / rel, current / rel
            base = {"file": rel, "path": "", "rule": f"(files: {cls})", "check": cls, "ref": None, "new": None,
                    "delta": None, "tol": None, "gate": True}
            if not bp.exists() or not cp.exists():
                if cls == "font_dependent" and not bp.exists():
                    f, ptt = {**base, "gate": False, "status": "REPORT",
                              "detail": "기준본(스냅샷)에 파일이 없어 비교를 생략함 — 보고만"}, tt
                elif optional:
                    f, ptt = {**base, "gate": False, "status": "N/A",
                              "detail": "선택 파일(ckpt 산출물) — " +
                                        ("기준본에 없음" if not bp.exists() else "재생성 결과에 없음")}, "선택 파일(없으면 N/A)"
                else:
                    f, ptt = {**base, "status": "FAIL",
                              "detail": "기준본에 파일 없음" if not bp.exists() else "재생성 결과에 파일 없음"}, "파일 있음"
                file_findings.append(f)
                _add_row(rows, f, "files", "presence", ptt)
                continue
            fc["files"] += 1
            try:
                if cls == "csv_cells":
                    f, c = _compare_csv(rel, bp, cp, rows=rows)
                    file_findings += f
                    fc["cells"] += c["cells"]
                    fc["identical_cells"] += c["identical"]
                elif cls in ("font_dependent", "pixel_equal"):
                    r = png_compare(bp, cp)
                    fc["png"] += 1
                    fc["png_pixel_equal"] += int(r["pixel_equal"])
                    if r["pixel_equal"]:
                        same = {**base, "gate": False, "status": "REPORT" if cls == "font_dependent" else "PASS",
                                "data": r, "detail": "RGBA 픽셀 동일"}
                        _add_row(rows, same, "files", "png", tt, identical=True)
                    else:
                        if cls == "font_dependent":
                            f = {**base, "gate": False, "status": "REPORT", "data": r,
                                 "detail": f"폰트 의존 그림 차이(비율 {r['changed_ratio']})"}
                        else:
                            f = {**base, "status": "FAIL", "data": r, "detail": "픽셀 불일치"}
                        file_findings.append(f)
                        _add_row(rows, f, "files", "png", tt)
                elif cls == "exact":
                    if _norm_sha256(bp) != _norm_sha256(cp):
                        if on_diff == "report":       # 보고만(FAIL 아님), 단 PASS_EXACT는 막는다(gate True)
                            f = {**base, "status": "REPORT", "detail": "LF 정규화 sha256 불일치 — 보고만, PASS_EXACT는 아님"}
                        else:
                            f = {**base, "status": "FAIL", "detail": "LF 정규화 sha256 불일치"}
                        file_findings.append(f)
                        _add_row(rows, f, "files", "sha256", tt)
                    else:
                        _add_row(rows, {**base, "gate": False, "status": "REPORT" if on_diff == "report" else "PASS",
                                        "detail": "LF 정규화 sha256 일치"}, "files", "sha256", tt, identical=True)
                else:
                    f = {**base, "status": "FAIL", "detail": f"알 수 없는 class: {cls}"}
                    file_findings.append(f)
                    _add_row(rows, f, "files", "error", tt)
            except Exception as e:
                f = {**base, "status": "FAIL", "detail": f"비교 중 오류: {e!r}"}
                file_findings.append(f)
                _add_row(rows, f, "files", "error", tt)

    def by_status(items):
        return dict(collections.Counter(f["status"] for f in items))

    counts = {"mode": mode, "json": {**jc, "status": by_status(findings)},
              "files": {**fc, "status": by_status(file_findings)},
              "status": by_status(findings + file_findings)}
    out = {"findings": findings + file_findings, "counts": counts}
    if rows is not None:
        out["rows"] = rows
    return out


def _cap(v):
    if v is None:
        return None
    if isinstance(v, str):
        return tuple(int(x) for x in re.findall(r"\d+", v))
    try:
        return tuple(int(x) for x in v)
    except TypeError:
        return (int(v),)


def _device_matches(device: dict, reference: dict) -> bool:
    if not device or not reference:
        return False
    for k, want in reference.items():
        got = device.get(k, _MISSING)
        if got is _MISSING:
            return False
        if k == "capability":
            if _cap(got) != _cap(want):
                return False
        elif k == "device":
            if str(got).split(":")[0].lower() != str(want).split(":")[0].lower():
                return False
        elif _num(want) is not None:
            g = _num(got)
            if g is None and isinstance(got, str):
                try:
                    g = float(got)
                except ValueError:
                    return False
            if g != _num(want):
                return False
        elif str(got) != str(want):
            return False
    return True


def regen_mode(device: dict, reference: dict) -> str:
    """device_info()가 tolerances.yaml의 reference_device와 같으면 'ref', 아니면 'other'."""
    return "ref" if _device_matches(device, reference) else "other"


def regen_label(result: dict, device: dict, reference: dict) -> str:
    """FAIL이 하나라도 있으면 FAIL. 장치가 reference와 같고 게이트 대상 비동일 항목이 0이면
    PASS_EXACT(purity·nmi 등 gate=False인 보고 항목은 무시). 그 밖에는 PASS_TOLERANCE. result["findings"]만 본다."""
    allf = list(result.get("findings", []))
    if any(f.get("status") == "FAIL" for f in allf):
        return "FAIL"
    if _device_matches(device, reference) and not any(f.get("gate", True) for f in allf):
        return "PASS_EXACT"
    return "PASS_TOLERANCE"


# ============================================================================ compare(재학습)

def t_factor(alpha: float, m: int, df: int) -> float:
    """Bonferroni 양측 t 분위수: scipy.stats.t.ppf(1 - alpha/(2m), df)."""
    from scipy import stats

    return float(stats.t.ppf(1.0 - alpha / (2.0 * m), df))


class _Tree:
    """기준본·현재 루트의 JSON 캐시."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._cache: dict = {}

    def json(self, rel: str):
        return _load_json_cached(self.root / rel, self._cache)

    def metrics(self, exp: str):
        return self.json(f"results/{exp}/metrics.json")


def _parse_time(v):
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    try:
        s = str(v).strip().replace("Z", "+00:00")
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


def _start_of(rec) -> float | None:
    """상태 레코드의 시작 시각(초). started_ns가 있으면 그것을, 없으면 ISO started를 쓴다."""
    if not isinstance(rec, dict):
        return None
    ns = rec.get("started_ns")
    if isinstance(ns, int) and not isinstance(ns, bool):
        return ns / 1e9
    return _parse_time(rec.get("started"))


def _trained_from_status(status) -> list[str]:
    if not isinstance(status, dict):
        return []
    names = list(status.get("trained") or [])
    for it in status.get("items") or []:
        if isinstance(it, dict):
            names += list(it.get("trained") or [])
    return list(dict.fromkeys(str(n) for n in names))


def _sha256_raw(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _status_runs(status) -> list[dict]:
    """실행 단위 상태 목록(시작 순). nmfc_cli.merged_status가 붙인 status['runs']가 있으면 그것, 없으면 [status]."""
    if not isinstance(status, dict):
        return []
    runs = status.get("runs")
    if isinstance(runs, list) and runs:
        return [r for r in runs if isinstance(r, dict)]
    return [status]


def _run_failed(run: dict) -> bool:
    """pass가 True가 아니면(False·None·누락 = 실패·중단·진행 중) 실패. exit_code가 있으면 0이어야 한다."""
    if run.get("pass") is not True:
        return True
    ec = run.get("exit_code")
    return ec is not None and ec != 0


def _failed_items(run: dict) -> list[dict]:
    """필수 항목 중 실패(실행했고 ok가 True가 아님)이거나 시작했으나 끝나지 않은(중단) 항목."""
    out = []
    for it in run.get("items") or []:
        if not isinstance(it, dict) or it.get("required") is False:
            continue
        ran = it.get("exit") is not None
        inflight = not ran and not it.get("skipped_reason") and it.get("started_ns") is not None
        if (ran and it.get("ok") is not True) or inflight:
            out.append(it)
    return out


def _review_runs(runs: list[dict]) -> tuple[list[dict], list[str]]:
    """실행별 요약과 전제조건 사유. 마지막 실행은 성공(pass True, exit_code 0)해야 하고, 그 앞의 실패 실행은
    실패 항목이 뒤의 성공 항목으로 해소되어야 한다(학습: 끝내지 못한 런이 뒤의 ok 학습 항목의 trained에 있음,
    단계: 같은 id가 뒤에서 ok). 학습 없이 성공한 마지막 실행(예: reproduce --from S3)은 막지 않는다."""
    summaries, reasons = [], []
    for i, run in enumerate(runs):
        later_ok = [it for r in runs[i + 1:] for it in (r.get("items") or [])
                    if isinstance(it, dict) and it.get("ok") is True]
        later_trained = {str(e) for it in later_ok if it.get("kind") == "train" for e in (it.get("trained") or [])}
        later_ids = {str(it.get("id")) for it in later_ok}
        failed, unresolved = [], []
        for it in _failed_items(run):
            iid = str(it.get("id"))
            if it.get("kind") == "train" and isinstance(it.get("expected"), list):
                skip = set(map(str, it.get("cached") or [])) | set(map(str, it.get("trained") or []))
                todo = [str(e) for e in it["expected"] if str(e) not in skip]
                failed.append({"id": iid, "kind": "train", "exit": it.get("exit"), "unfinished": todo})
                left = [e for e in todo if e not in later_trained]
                if left:
                    unresolved.append(f"{iid} {left}")
            else:
                failed.append({"id": iid, "kind": it.get("kind"), "exit": it.get("exit")})
                if iid not in later_ids:
                    unresolved.append(iid)
        expected = [str(e) for it in (run.get("items") or []) if isinstance(it, dict) and it.get("kind") == "train"
                    for e in (it.get("expected") or []) if e not in (it.get("cached") or [])]
        is_latest = i == len(runs) - 1
        run_failed = _run_failed(run)
        if not run_failed:
            state = "ok"
        elif is_latest:
            state = "latest_failed"
        else:
            state = "unresolved" if unresolved else "resolved"
        what = ", ".join(f"{f['id']}(exit {f['exit']}" + (f", 미완료 {f['unfinished']}" if f.get("unfinished") else "")
                         + ")" for f in failed) or "실패 항목 기록 없음(pass≠True 또는 exit_code≠0)"
        when = f"시작 {run.get('started')}, pass={run.get('pass')}, exit_code={run.get('exit_code')}"
        if state == "latest_failed":
            reasons.append(f"마지막 train/reproduce 실행이 성공하지 않음({when}): {what} — 이 실행을 다시 성공시킨 뒤 비교")
        elif state == "unresolved":
            reasons.append(f"이전 실행의 실패가 이후 성공한 실행으로 해소되지 않음({when}): {', '.join(unresolved)}"
                           " — 그 런을 다시 학습하거나 --latest-only")
        summaries.append({"source": run.get("_source"), "started": run.get("started"), "finished": run.get("finished"),
                          "pass": run.get("pass"), "exit_code": run.get("exit_code"), "args": run.get("args"),
                          "trained": [str(e) for e in run.get("trained") or []], "expected_to_train": expected,
                          "failed_items": failed, "unresolved": unresolved, "state": state, "latest": is_latest})
    return summaries, reasons


def _artifact_checks(exp: str, fp, mp: Path, npz: Path, base_mp: Path, t_exp, slack: float):
    """학습 산출물 신선도. (전제조건 사유 목록, 보고 목록).

    지문(파이프라인이 학습 직후 기록한 원시 sha256·size)이 있으면 지금 파일이 그와 같아야 한다. 없으면(이전 형식
    상태) 보조 규칙: metrics.json이 기준본과 LF 정규화 내용이 같으면 이번 학습 산출물로 보지 않고, embeddings.npz가
    실행 시작보다 오래됐으면 다시 쓰이지 않은 것으로 본다."""
    reasons, reports = [], []
    fp = fp if isinstance(fp, dict) else {}
    mrec, nrec = fp.get("metrics.json"), fp.get("embeddings.npz")
    has_fp = isinstance(mrec, dict) and bool(mrec.get("sha256"))

    def differs(path: Path, rec: dict) -> bool:          # size는 기록돼 있을 때만 본다(sha256이 판정 기준)
        if rec.get("size") is not None and path.stat().st_size != rec["size"]:
            return True
        return _sha256_raw(path) != rec["sha256"]

    if has_fp:
        if differs(mp, mrec):
            reasons.append(f"{exp}: metrics.json이 학습 직후 기록(sha256 {str(mrec['sha256'])[:12]}…)과 다름")
        elif base_mp.is_file() and _norm_sha256(base_mp) == _norm_sha256(mp):
            reports.append({"kind": "identical", "exp": exp, "metric": "metrics.json", "status": "WARN",
                            "detail": "metrics.json 내용(LF 정규화 sha256)이 기준본과 같고, 학습 직후 기록한 sha256과도 같음"})
    elif base_mp.is_file() and _norm_sha256(base_mp) == _norm_sha256(mp):
        reasons.append(f"{exp}: metrics.json이 기준본과 내용이 같고 학습 산출물 지문(items[*].artifacts)이 없음")
    if isinstance(nrec, dict) and nrec.get("sha256"):
        if not npz.is_file():
            reasons.append(f"{exp}: embeddings.npz가 학습 직후에는 있었으나 지금은 없음")
        elif differs(npz, nrec):
            reasons.append(f"{exp}: embeddings.npz가 학습 직후 기록(sha256 {str(nrec['sha256'])[:12]}…)과 다름")
        if nrec.get("rewritten") is False:
            reasons.append(f"{exp}: embeddings.npz 지문의 rewritten이 False(학습 전 파일)")
    elif has_fp and "embeddings.npz" in fp and nrec is None and npz.is_file():
        reasons.append(f"{exp}: embeddings.npz가 학습 직후에는 없었는데 지금은 있음")
    elif not has_fp and npz.is_file() and t_exp is not None and npz.stat().st_mtime < t_exp - slack:
        reasons.append(f"{exp}: embeddings.npz mtime이 실행 시작보다 이전임(여유 {slack:g}s)")
    return reasons, reports


def _is_mount(path: Path) -> bool:
    p = Path(path)
    if os.path.ismount(p):
        return True
    try:
        target = str(p.resolve())
        with open("/proc/self/mountinfo", encoding="utf-8") as f:
            return any(line.split()[4] == target for line in f if len(line.split()) > 4)
    except OSError:
        return False


def _family_index(base: _Tree, rt: dict):
    """exp → {family, group, seed, kind} 와 (family, group) → {seed: exp}."""
    idx, groups = {}, {}
    for fname, f in (rt.get("families") or {}).items():
        kind = f.get("kind", "seeded")
        if "roles" in f:
            seeds = [str(s) for s in f.get("seeds", [42])]
            base_seed = str(f.get("base_seed", 42))
            suffix = f.get("seed_suffix", "_s{seed}")
            for role, bexp in f["roles"].items():
                for s in seeds:
                    exp = bexp if s == base_seed else bexp + suffix.format(seed=s)
                    idx[exp] = {"family": fname, "group": role, "seed": s, "kind": kind}
                    groups.setdefault((fname, role), {})[s] = exp
        elif "cells_from" in f:
            cf = f["cells_from"]
            rows = _resolve(base.json(cf["file"]), _split(cf.get("rows", "rows")), None)
            if not isinstance(rows, list):
                continue
            for r in rows:
                if not isinstance(r, dict):
                    continue
                cell = "/".join(str(r.get(k)) for k in cf["cell"])
                s, exp = str(r.get(cf.get("seed", "seed"))), str(r.get(cf.get("exp", "exp")))
                idx[exp] = {"family": fname, "group": cell, "seed": s, "kind": kind}
                groups.setdefault((fname, cell), {})[s] = exp
    return idx, groups


def _source_value(tree: _Tree, mspec: dict, ctx: dict):
    if mspec.get("source", "metrics") == "metrics":
        m = tree.metrics(ctx["exp"])
        return _MISSING if m is _MISSING else _dotted(m, mspec["key"])
    obj = tree.json(mspec["source"])
    return _resolve(obj, _split(str(mspec["path"]).format(**ctx)), ROWS_KEY)


def _source_fresh(base: _Tree, cur: _Tree, mspec: dict, exp: str, hcache: dict):
    """(신선?, 사유). metrics.json 출처는 런 단위 전제조건으로 이미 확인했다."""
    src = mspec.get("source", "metrics")
    if src == "metrics":
        return True, ""
    cp = cur.root / src
    if not cp.exists():
        return False, f"{src} 없음"
    npz = cur.root / "results" / exp / "embeddings.npz"
    ref_file = npz if npz.exists() else cur.root / "results" / exp / "metrics.json"
    if ref_file.exists() and cp.stat().st_mtime_ns < ref_file.stat().st_mtime_ns:
        return False, f"{src} mtime이 {exp}/{ref_file.name} mtime보다 오래됨"
    bp = base.root / src
    if bp.exists():
        key = src
        if key not in hcache:
            hcache[key] = _norm_sha256(bp) == _norm_sha256(cp)
        if hcache[key]:
            return False, f"{src}가 기준본과 내용이 같음(LF 정규화 sha256 동일)"
    return True, ""


def _change_report(base_root: Path, cur_root: Path, trained: set, rt: dict) -> list[dict]:
    exts = (".json", ".md", ".csv")
    allowed = list(rt.get("expected_changes") or [])
    agg_pats = list(rt.get("aggregate_patterns") or [])

    def collect(root: Path):
        out = set()
        res = root / "results"
        if res.is_dir():
            for p in res.rglob("*"):
                if p.is_file() and p.suffix.lower() in exts:
                    out.add(p.relative_to(root).as_posix())
        logs = root / "logs"
        if logs.is_dir():
            for p in logs.glob("*/history.json"):
                out.add(p.relative_to(root).as_posix())
        return out

    bfiles, cfiles = collect(base_root), collect(cur_root)
    run_dirs = set()
    if (base_root / "results").is_dir():
        run_dirs = {d.name for d in (base_root / "results").iterdir() if (d / "metrics.json").exists()}
    reports = []
    for rel in sorted(bfiles | cfiles):
        if rel in bfiles and rel in cfiles:
            if _norm_sha256(base_root / rel) == _norm_sha256(cur_root / rel):
                continue
            change = "changed"
        else:
            change = "missing" if rel in bfiles else "new"
        parts = rel.split("/")
        exp = parts[1] if len(parts) >= 3 else None
        if exp in trained:
            cat = "trained"
        elif any(fnmatch.fnmatchcase(rel, p) for p in allowed):
            cat = "allowed"
        elif exp in run_dirs and not any(fnmatch.fnmatchcase(rel, p) for p in agg_pats):
            cat = "unexpected"
        else:
            cat = "aggregate"
        status = "WARN" if (cat == "unexpected" or change == "missing") else "REPORT"
        reports.append({"kind": "file_change", "path": rel, "change": change, "category": cat,
                        "status": status, "detail": f"{change} ({cat})"})
    return reports


def retrain_compare(baseline: Path, current: Path, status: dict, tol: dict) -> dict:
    """재학습 비교 판정(nmfc compare). 반환 {"gates","reports","unevaluated","pass","exit_code","precondition",
    "trained","summary","runs"}. exit_code: 0 PASS, 1 FAIL(게이트 실패), 2 전제조건 불충족·미평가·평가 게이트 0건."""
    rt = tol.get("retrain", tol) if isinstance(tol, dict) else {}
    baseline, current = Path(baseline), Path(current)
    out = {"gates": [], "reports": [], "unevaluated": [], "pass": False, "exit_code": 2,
           "precondition": {"ok": False, "reasons": []}, "trained": [], "summary": {}, "runs": []}
    trained = _trained_from_status(status)
    out["trained"] = trained

    # ---- 1. 전제조건
    reasons = []
    if not trained:
        reasons.append("runs/pipeline_status.json에 이번에 실제로 학습한 실험이 없음")
    runs, run_reasons = _review_runs(_status_runs(status))
    out["runs"] = runs
    reasons += run_reasons
    start = _start_of(status)
    item_start = {}  # 실험 → 그 실험을 학습한 항목의 시작 시각(있으면 이것이 더 정확하다)
    fprints = {}     # 실험 → 그 실험을 학습한 (마지막) 항목이 남긴 산출물 지문
    for it in (status.get("items") or []) if isinstance(status, dict) else []:
        if not isinstance(it, dict):
            continue
        t_it = _start_of(it)
        arts = it.get("artifacts") if isinstance(it.get("artifacts"), dict) else {}
        for e in it.get("trained") or []:
            if t_it is not None:
                item_start[str(e)] = t_it
            fprints[str(e)] = arts.get(e)        # 지문 없는 항목이 나중에 학습했으면 앞 지문을 지운다
    if trained and start is None and not item_start:
        reasons.append("pipeline_status.json에 실행 시작 시각(started)이 없음")
    slack = float(rt.get("mtime_slack_s", 1.0))
    pre_reports = []
    for exp in trained:
        mp = current / "results" / exp / "metrics.json"
        t_exp = item_start.get(exp, start)
        if not mp.exists():
            reasons.append(f"{exp}: results/{exp}/metrics.json 없음")
            continue
        if t_exp is not None and mp.stat().st_mtime < t_exp - slack:
            reasons.append(f"{exp}: metrics.json mtime이 실행 시작보다 이전임(여유 {slack:g}s)")
        r_, p_ = _artifact_checks(exp, fprints.get(exp), mp, current / "results" / exp / "embeddings.npz",
                                  baseline / "results" / exp / "metrics.json", t_exp, slack)
        reasons += r_
        pre_reports += p_
    if rt.get("require_volume_in_container", True) and os.environ.get("NMFC_IN_CONTAINER") == "1":
        if not _is_mount(current / "results"):
            reasons.append("컨테이너에서 results/가 볼륨(마운트)이 아님")
    out["precondition"] = {"ok": not reasons, "reasons": reasons}
    if reasons:
        out["reports"].extend(pre_reports)
        out["summary"] = {"exit_code": 2, "reason": "전제조건 불충족", "trained": len(trained)}
        return out

    base, cur = _Tree(baseline), _Tree(current)
    trained_set = set(trained)
    families = rt.get("families") or {}
    idx, groups = _family_index(base, rt)
    alpha = float(rt.get("alpha", 0.05))
    df_single = int((rt.get("single_run") or {}).get("df", 4))
    sd_mult = float(rt.get("report_sd_multiple", 4.2))
    rep_abs = float((rt.get("single_seed") or {}).get("report_abs", 0.02))
    syn = rt.get("synthetic") or {}
    te = rt.get("transition_epoch") or {}
    gates, reports, uneval = out["gates"], out["reports"], out["unevaluated"]
    reports.extend(pre_reports)
    hcache: dict = {}
    run_vals: dict = {}  # exp -> metric -> {"ref","new","ok","reason"}

    def gate(**kw):
        gates.append(kw)

    def report(**kw):
        kw.setdefault("status", "REPORT")
        reports.append(kw)

    def collect(exp, info, metric, mspec, as_gate=True):
        ctx = {"exp": exp, "role": info["group"], "seed": info["seed"]}
        rv = _source_value(base, mspec, ctx)
        fresh, why = _source_fresh(base, cur, mspec, exp, hcache)
        nv = _source_value(cur, mspec, ctx) if fresh else _MISSING
        src = mspec.get("source", "metrics")
        rec = {"ref": _num(rv), "new": _num(nv), "ok": False, "reason": "", "source": src}
        if rec["ref"] is None:
            rec["reason"] = "기준본 값 없음"
        elif not fresh:
            rec["reason"] = why
        elif rec["new"] is None:
            rec["reason"] = f"{src}에 값 없음"
        else:
            rec["ok"] = True
        if not rec["ok"]:
            item = {"exp": exp, "metric": metric, "source": src, "reason": rec["reason"]}
            if as_gate:
                uneval.append(item)
            else:
                report(kind="single_seed", exp=exp, metric=metric, status="N/A", detail=rec["reason"], source=src)
        return rec

    # ---- 2. 런별 판정
    for exp in trained:
        bm, cm = base.metrics(exp), cur.metrics(exp)
        if bm is _MISSING or cm is _MISSING:
            uneval.append({"exp": exp, "metric": "*", "source": f"results/{exp}/metrics.json",
                           "reason": "기준본 metrics.json 없음" if bm is _MISSING else "metrics.json 읽기 실패"})
            continue
        covered = set()
        for key in rt.get("exact_keys") or []:
            a = _dotted(bm, key)
            if a is _MISSING:
                continue
            b = _dotted(cm, key)
            covered.add(key.split(".")[0])
            gate(kind="exact", exp=exp, metric=key, ref=_brief(a), new=_brief(b),
                 status="PASS" if _same(a, b) else "FAIL", detail="런별 정확 일치(기준본 값)")
        for key in rt.get("config_keys") or []:
            a = _dotted(bm, key)
            if a is _MISSING:
                continue
            b = _dotted(cm, key)
            covered.add(key.split(".")[0])
            if not _same(a, b):
                report(kind="config", exp=exp, metric=key, ref=_brief(a), new=_brief(b), status="WARN",
                       detail="설정 값이 기준본과 다름")
        if "transition_epoch" in bm:
            covered.add("transition_epoch")
            a, b = bm.get("transition_epoch"), cm.get("transition_epoch", _MISSING)
            apt = (bm.get("variant") or {}).get("apt") if isinstance(bm.get("variant"), dict) else None
            code = bm.get("variant_code")
            if code in (te.get("report_variant_codes") or []):
                report(kind="transition_epoch", exp=exp, metric="transition_epoch", ref=a, new=_brief(b),
                       transition_reason=cm.get("transition_reason"),
                       detail=f"variant_code {code}(report_variant_codes): 보고만")
            elif apt in (te.get("exact_apt") or ["reciprocal"]):
                gate(kind="exact", exp=exp, metric="transition_epoch", ref=a, new=_brief(b),
                     status="PASS" if _same(a, b) else "FAIL",
                     detail=f"variant.apt={apt}: 정확 일치")
            else:
                report(kind="transition_epoch", exp=exp, metric="transition_epoch", ref=a, new=_brief(b),
                       transition_reason=cm.get("transition_reason"), detail="apt 정보 없음: 보고만")

        info = idx.get(exp)
        is_syn = bm.get("runner") == syn.get("runner", "synthetic") or (
            syn.get("prefix") and exp.startswith(syn["prefix"]))
        if is_syn and syn.get("metrics"):
            for metric, t in syn["metrics"].items():
                covered.add(metric)
                a, b = _num(_dotted(bm, metric)), _num(_dotted(cm, metric))
                if a is None or b is None:
                    uneval.append({"exp": exp, "metric": metric, "source": "metrics",
                                   "reason": "metrics.json에 값 없음"})
                    continue
                d = abs(b - a)
                gate(kind="synthetic", exp=exp, metric=metric, ref=a, new=b, delta=d, threshold=float(t),
                     status="PASS" if _within(d, float(t)) else "FAIL", detail=f"허용오차 ±{t}(retrain.synthetic.metrics)")
        elif info and families.get(info["family"], {}).get("kind", "seeded") == "seeded":
            fam = families[info["family"]]
            run_vals[exp] = {}
            for metric, mspec in (fam.get("metrics") or {}).items():
                if mspec.get("source", "metrics") == "metrics":
                    covered.add(mspec["key"].split(".")[0])
                run_vals[exp][metric] = collect(exp, info, metric, mspec, as_gate=True)
        elif info:  # 단일 시드 가족(DML 4종 등): 보고만
            fam = families[info["family"]]
            for metric, mspec in (fam.get("metrics") or {}).items():
                if mspec.get("source", "metrics") == "metrics":
                    covered.add(mspec["key"].split(".")[0])
                rec = collect(exp, info, metric, mspec, as_gate=False)
                if rec["ok"]:
                    d = abs(rec["new"] - rec["ref"])
                    report(kind="single_seed", exp=exp, metric=metric, ref=rec["ref"], new=rec["new"], delta=d,
                           threshold=rep_abs, status="WARN" if d > rep_abs else "REPORT",
                           detail=f"분산 미상 단일 시드: |Δ|{'>' if d > rep_abs else '≤'}{rep_abs}")
        if not info and not is_syn:
            for es in (rt.get("single_seed") or {}).get("extra_sources") or []:
                ctx = {"exp": exp, "role": None, "seed": None}
                if _source_value(base, es, ctx) is _MISSING:
                    continue
                rec = collect(exp, {"group": None, "seed": None}, es["name"], es, as_gate=False)
                if rec["ok"]:
                    d = abs(rec["new"] - rec["ref"])
                    report(kind="single_seed", exp=exp, metric=es["name"], ref=rec["ref"], new=rec["new"], delta=d,
                           threshold=rep_abs, status="WARN" if d > rep_abs else "REPORT",
                           detail=f"분산 미상 단일 시드: |Δ|{'>' if d > rep_abs else '≤'}{rep_abs}")
        # 그 밖의 float 지표: 분산 미상 → |Δ| > 0.02 보고만
        for key, a in bm.items():
            if key in covered or not _is_float(a):
                continue
            b = cm.get(key, _MISSING)
            if not _is_float(b):
                report(kind="single_seed", exp=exp, metric=key, ref=a, new=_brief(b), status="WARN",
                       detail="float 지표가 사라지거나 형이 바뀜")
                continue
            d = abs(b - a)
            report(kind="single_seed", exp=exp, metric=key, ref=a, new=b, delta=d, threshold=rep_abs,
                   status="WARN" if d > rep_abs else "REPORT",
                   detail=f"분산 미상: |Δ|{'>' if d > rep_abs else '≤'}{rep_abs} (보고만)")

    # ---- 3. 역할(셀) 단위: 단일 런 게이트 또는 평균 게이트
    sd_cache: dict = {}

    def base_values(fname, group, metric):
        fam = families[fname]
        mspec = fam["metrics"][metric]
        vals = {}
        for s, e in groups[(fname, group)].items():
            v = _num(_source_value(base, mspec, {"exp": e, "role": group, "seed": s}))
            if v is not None:
                vals[s] = v
        return vals

    def base_sd(fname, group, metric):
        key = (fname, group, metric)
        if key not in sd_cache:
            vals = base_values(fname, group, metric)
            ok = len(vals) == len(groups[(fname, group)]) and len(vals) >= 2
            sd_cache[key] = statistics.stdev(vals.values()) if ok else None
        return sd_cache[key]

    for (fname, group), seedmap in groups.items():
        fam = families.get(fname) or {}
        if fam.get("kind", "seeded") != "seeded":
            continue
        tseeds = [s for s, e in seedmap.items() if e in trained_set]
        if not tseeds:
            continue
        metrics = list((fam.get("metrics") or {}).keys())
        m_run = max(1, len(metrics))
        all_seeds = len(tseeds) == len(seedmap)
        n = len(seedmap)
        if all_seeds and int(fam.get("n", n)) != n:
            report(kind="config", family=fname, group=group, status="WARN",
                   detail=f"가족 n={fam.get('n')}과 실제 시드 수 {n}이 다름")
        k_single = t_factor(alpha, m_run, df_single) * math.sqrt(2.0)
        for metric in metrics:
            s = base_sd(fname, group, metric)
            if s is None:
                for sd_seed in tseeds:
                    uneval.append({"exp": seedmap[sd_seed], "metric": metric, "source": "baseline",
                                   "reason": "기준본 시드 값이 모자라 SD를 계산할 수 없음"})
                continue
            for sd_seed in tseeds:
                exp = seedmap[sd_seed]
                rec = run_vals.get(exp, {}).get(metric)
                if not rec or not rec["ok"]:
                    continue
                d = abs(rec["new"] - rec["ref"])
                thr = k_single * s
                ok = _within(d, thr)
                entry = dict(kind="single_run", family=fname, group=group, exp=exp, seed=sd_seed, metric=metric,
                             ref=rec["ref"], new=rec["new"], delta=d, s=s, threshold=thr, m_run=m_run,
                             df=df_single)
                if all_seeds:
                    report(**entry, status="REPORT" if ok else "WARN",
                           detail=f"{n}시드 전부 재학습: 런별 단일 런 기준은 보고만")
                else:
                    gate(**entry, status="PASS" if ok else "FAIL",
                         detail=f"|Δ| ≤ t(1-{alpha}/(2·{m_run}), df={df_single})·√2·s")
                over = d > sd_mult * s
                report(kind="sd_multiple", family=fname, group=group, exp=exp, metric=metric, delta=d, s=s,
                       threshold=sd_mult * s, status="WARN" if over else "REPORT",
                       detail=f"런별 {sd_mult}·SD {'초과' if over else '이내'}(보고만)")
            if all_seeds:
                recs = [run_vals.get(seedmap[x], {}).get(metric) for x in seedmap]
                if not all(r and r["ok"] for r in recs):
                    uneval.append({"family": fname, "group": group, "metric": metric, "gate": "mean",
                                   "reason": "일부 시드의 지표가 미평가라 평균 게이트를 계산할 수 없음"})
                    continue
                mean_new = statistics.fmean(r["new"] for r in recs)
                mean_ref = statistics.fmean(r["ref"] for r in recs)
                m = int(fam.get("m", 20))
                thr = t_factor(alpha, m, n - 1) * s * math.sqrt(2.0 / n)
                d = abs(mean_new - mean_ref)
                gate(kind="mean", family=fname, group=group, metric=metric, ref=mean_ref, new=mean_new, delta=d,
                     s=s, threshold=thr, n=n, m=m, status="PASS" if _within(d, thr) else "FAIL",
                     detail=f"|mean_new−mean_ref| ≤ t(1-{alpha}/(2·{m}), df={n - 1})·s·√(2/{n})")

    # ---- 4. 부호 게이트·보고
    sg = rt.get("sign_gate") or {}
    ratio_min = float(sg.get("ratio_min", 3.0))

    def sign_eval(entry, as_gate):
        fname = entry.get("family", sg.get("family", "core"))
        fam = families.get(fname) or {}
        ra, rb = entry["a"], entry["b"]
        if (fname, ra) not in groups or (fname, rb) not in groups:
            return
        sa, sb = groups[(fname, ra)], groups[(fname, rb)]
        seeds = [s for s in sa if s in sb]
        for metric in entry.get("metrics") or []:
            if metric not in (fam.get("metrics") or {}):
                continue
            involved = [x[s] for s in seeds for x in (sa, sb) if x[s] in trained_set]
            if not involved:
                continue
            bva, bvb = base_values(fname, ra, metric), base_values(fname, rb, metric)
            d_ref, d_new, missing = {}, {}, []
            for s in seeds:
                if s not in bva or s not in bvb:
                    missing.append(f"기준본 {s}")
                    continue
                vals = []
                for role_seedmap, bv in ((sa, bva), (sb, bvb)):
                    e = role_seedmap[s]
                    if e in trained_set:
                        rec = run_vals.get(e, {}).get(metric)
                        if not rec or not rec["ok"]:
                            missing.append(e)
                            vals.append(None)
                            continue
                        vals.append(rec["new"])
                    else:
                        vals.append(bv[s])
                d_ref[s] = bva[s] - bvb[s]
                if None not in vals:
                    d_new[s] = vals[0] - vals[1]
            name = f"{ra}_vs_{rb}/{metric}"
            if missing:
                if as_gate:
                    uneval.append({"family": fname, "group": name, "metric": metric, "gate": "sign",
                                   "reason": f"값 미평가: {sorted(set(missing))}"})
                else:
                    report(kind="sign", family=fname, group=name, metric=metric, status="N/A",
                           detail=f"값 미평가: {sorted(set(missing))}")
                continue
            n = len(d_ref)
            pos_ref = sum(1 for v in d_ref.values() if v > 0)
            pos_new = sum(1 for v in d_new.values() if v > 0)
            ratio = (abs(statistics.fmean(d_ref.values())) / statistics.stdev(d_ref.values())
                     if n >= 2 and statistics.stdev(d_ref.values()) > 0 else None)
            entry_out = dict(kind="sign", family=fname, group=name, metric=metric, ref=f"{pos_ref}/{n}",
                             new=f"{pos_new}/{n}", ratio=ratio, involved=involved)
            if as_gate:
                if pos_ref not in (0, n):
                    report(**entry_out, status="WARN", detail="기준본 부호가 전부 같지 않아 유지 게이트를 정의할 수 없음")
                    continue
                gate(**entry_out, status="PASS" if pos_new == pos_ref else "FAIL",
                     detail=f"부호 {pos_ref}/{n} 유지(|mean|/SD={ratio:.2f})" if ratio is not None else "부호 유지")
                if ratio is not None and ratio < ratio_min:
                    report(kind="sign_ratio", family=fname, group=name, metric=metric, ratio=ratio, status="WARN",
                           detail=f"목록 지표의 |mean|/SD={ratio:.2f} < {ratio_min}")
            else:
                report(**entry_out, status="REPORT", detail="부호 개수는 보고만")

    for e in sg.get("gate") or []:
        sign_eval(e, True)
    for e in sg.get("report") or []:
        sign_eval(e, False)

    # ---- 5. 파일 변화 보고(retrain.expected_changes·aggregate_patterns)
    try:
        reports.extend(_change_report(baseline, current, trained_set, rt))
    except OSError as e:
        report(kind="file_change", status="WARN", detail=f"파일 변화 집계 실패: {e!r}")

    # ---- 6. 판정
    fail = any(g["status"] == "FAIL" for g in gates)
    if fail:
        code = 1
    elif uneval and rt.get("fail_on_unevaluated", True):
        code = 2
    elif not gates:
        code = 2
    else:
        code = 0
    out["exit_code"] = code
    out["pass"] = code == 0
    out["summary"] = {"exit_code": code, "trained": len(trained), "runs": len(out["runs"]),
                      "gates": dict(collections.Counter(g["status"] for g in gates)),
                      "reports": dict(collections.Counter(r["status"] for r in reports)),
                      "unevaluated": len(uneval),
                      "reason": ("게이트 실패" if fail else "미평가 지표 있음" if code == 2 and uneval else
                                 "평가된 게이트 0건" if code == 2 else "통과")}
    return out
