"""큐 대조.

scripts/run_all.ps1을 직접 펼친 (순서, config, only) 목록이 이식판 QUEUE(queue_items)와 같은지 단언한다.
foreach 값(0..3, 0..2)은 적힌 그대로 펼친다. 런 집합(127)은 독립 구현(줄 정규식·실재 config glob·집합)과 교차 확인하고,
그 127런이 모두 커밋된 원자료(results/<런>/metrics.json, logs/<런>/history.json)를 가졌는지 본다.
집합 방식은 순서·foreach 값 변경을 못 잡는다 — 그 차이를 망가뜨린 입력으로 보여 준다.
"""
from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path

import pytest

DOCKER = Path(__file__).resolve().parents[1]
REPO = DOCKER.parent
if str(DOCKER) not in sys.path:
    sys.path.insert(0, str(DOCKER))

import nmfc_pipeline as P  # noqa: E402

PS1 = REPO / "scripts" / "run_all.ps1"


def ps1_text() -> str:
    return PS1.read_text(encoding="utf-8")


def ps1_lines() -> list[str]:
    return ps1_text().split("\n")


def line_range(spec: str) -> range:
    a, _, b = spec.partition("-")
    return range(int(a), int(b or a) + 1)


def glob_set_queued(text: str, root: Path = REPO) -> set[str]:
    """P.parse_ps1_queue와 독립인 큐 해석(집합): 줄마다 --config 정규식, `$s`는 실재 config 파일 glob,
    런 이름은 config 원문의 `name:` 정규식, --only는 정의된 이름과 교집합. 순서·foreach 값은 보지 않는다."""
    text = text.replace("`\n", " ")
    name_re = re.compile(r"name:\s*([A-Za-z0-9_]+)")
    queued: set[str] = set()
    for line in text.splitlines():
        line = line.split("#", 1)[0]
        if "--config" not in line:
            continue
        m = re.search(r'--config\s+"?configs\\([A-Za-z0-9_.$]+\.yaml)"?', line)
        assert m, f"--config 정규식이 못 읽는 줄: {line.strip()}"
        raw = m.group(1)
        if "$" in raw:
            cfgs = sorted(glob.glob(os.path.join(root, "configs", re.sub(r"\$\w+", "*", raw))))
        else:
            cfgs = [os.path.join(root, "configs", raw)]
        only = re.search(r"--only\s+((?:[A-Za-z0-9_]+\s*)+)", line)
        for cfg in cfgs:
            defined = set(name_re.findall(open(cfg, encoding="utf-8").read()))
            queued |= (set(only.group(1).split()) & defined) if only else defined
    return queued


def port_runs() -> list[str]:
    return [n for cfg, only in P.queue_items() for n in P.experiments_in(cfg, only, REPO)]


# ---------------------------------------------------------------- 정상 입력

def test_ps1_unrolled_equals_port_queue():
    parsed = P.parse_ps1_queue(ps1_text())
    assert parsed == P.queue_items()
    assert len(parsed) == 27                                   # config 호출 27회
    assert P.queue_parity_errors(ps1_text()) == []


def test_queue_stage_shape():
    stages = [q["stage"] for q in P.QUEUE]
    assert stages == sorted(stages)                            # T1..T8 순서 그대로
    per_stage = {s: [q for q in P.QUEUE if q["stage"] == s] for s in P.STAGE_IDS}
    assert {s: len(v) for s, v in per_stage.items()} == {
        "T1": 1, "T2": 1, "T3": 1, "T4": 1, "T5": 7, "T6": 8, "T7": 2, "T8": 6}
    runs = {s: sum(len(P.experiments_in(q["config"], q["only"], REPO)) for q in v) for s, v in per_stage.items()}
    assert runs == {"T1": 9, "T2": 4, "T3": 1, "T4": 4, "T5": 15, "T6": 24, "T7": 16, "T8": 54}
    assert [q["config"] for q in per_stage["T5"]] == [
        "configs/cifar100_variants.yaml", "configs/cifar100_variants_wave2.yaml",
        "configs/cifar100_scratch.yaml", "configs/cifar100_dml_scratch.yaml",
        "configs/ood_cifar10_softmax_lr3.yaml", "configs/ood_cifar10_variants.yaml",
        "configs/cifar100_openset.yaml"]
    assert per_stage["T5"][3]["only"] == ["c100s_proxyanchor"]


def test_foreach_values_unrolled_literally_and_interleaved():
    parsed = P.parse_ps1_queue(ps1_text())
    seeded = [(Path(c).stem.rsplit("_s", 1)[0], Path(c).stem.rsplit("_s", 1)[1])
              for c, _ in parsed if re.search(r"_s\d+\.yaml$", c)]
    assert seeded == [(base, s) for s in "0123" for base in ("cifar100_seeds", "ood_cifar10_seeds")] + \
                     [(base, s) for s in "012" for base in ("cifar100_openset_seeds", "cifar100_openset_noise_grid")]
    loops = {e["foreach"]["lines"]: e["foreach"] for e in P._parse_ps1(ps1_text()) if "foreach" in e}
    assert sorted(loops) == ["41-44", "49-52"]
    vals = {span: tuple(sorted({e["foreach"]["value"] for e in P._parse_ps1(ps1_text())
                                if e.get("foreach", {}).get("lines") == span})) for span in loops}
    assert vals == {"41-44": ("0", "1", "2", "3"), "49-52": ("0", "1", "2")}
    assert P.STAGES["T6"]["foreach"] == {"var": "s", "values": ("0", "1", "2", "3")}
    assert P.STAGES["T8"]["foreach"] == {"var": "s", "values": ("0", "1", "2")}
    assert P.STAGES["T6"]["ps1_lines"] == "41-44" and P.STAGES["T8"]["ps1_lines"] == "49-52"


def test_ps1_line_numbers_are_measured():
    lines = ps1_lines()
    for q in P.QUEUE:
        seg = "\n".join(lines[i - 1] for i in line_range(q["ps1_lines"]))
        stem = re.sub(r"_s\d+$", "_s$s", Path(q["config"]).stem)    # foreach 본문은 보간 전 원문
        assert f"configs\\{stem}.yaml" in seg, (q, seg)
        for name in q["only"] or []:
            assert re.search(rf"(?<![A-Za-z0-9_]){name}(?![A-Za-z0-9_])", seg), (q, name)
    entries = [e for e in P._parse_ps1(ps1_text()) if e["kind"] == "train"]
    assert [e["lines"] for e in entries] == [q["ps1_lines"] for q in P.QUEUE]
    # 단계 범위: 항목 줄을 모두 덮고, 범위 양끝은 명령/루프 줄이다
    for sid, meta in P.STAGES.items():
        r = line_range(meta["ps1_lines"])
        item_lines = [n for q in P.QUEUE if q["stage"] == sid for n in line_range(q["ps1_lines"])]
        assert set(item_lines) <= set(r), sid
        assert lines[r.start - 1].strip() and lines[r.stop - 2].strip(), sid
    # 39행은 빈 줄이라 T5 범위는 :32-38이다
    assert P.STAGES["T5"]["ps1_lines"] == "32-38" and lines[39 - 1].strip() == ""


def test_run_set_is_127_and_equals_independent_glob_set():
    runs = port_runs()
    assert len(runs) == 127 and len(set(runs)) == 127          # 중복 없음
    assert set(runs) == glob_set_queued(ps1_text())
    # 큐의 127런은 모두 커밋된 원자료를 가진다
    missing = [n for n in runs if not (REPO / "results" / n / "metrics.json").is_file()
               or not (REPO / "logs" / n / "history.json").is_file()]
    assert missing == []


def test_experiments_in_follows_train_py():
    # config 순서(= src.train 실제 학습 순서)로 돌려준다 — --only 순서가 아니다
    t1 = P.QUEUE[0]
    assert P.experiments_in(t1["config"], t1["only"], REPO) == [
        "syn_moons_linear", "syn_circles_linear", "syn_swiss_linear", "syn_moons_softmax", "syn_moons_nmfc",
        "syn_circles_softmax", "syn_circles_nmfc", "syn_swiss_softmax", "syn_swiss_nmfc"]
    assert {"syn_moons_apt_none", "syn_moons_apt_fixed3"} <= set(
        P.experiments_in("configs/synthetic.yaml", None, REPO))   # --only가 뺀 런(results/에 없음)
    assert P.experiments_in("configs/base.yaml", None, REPO) == []
    for q in P.QUEUE:
        if q["only"]:
            assert sorted(P.experiments_in(q["config"], q["only"], REPO)) == sorted(q["only"])
    with pytest.raises(ValueError, match="정의되지 않은"):
        P.experiments_in("configs/cifar100.yaml", ["cifar100_nmfc", "typo_name"], REPO)


def test_crlf_and_bom_are_normalized():
    text = ps1_text()
    assert P.parse_ps1_queue(text.replace("\n", "\r\n")) == P.queue_items()
    assert P.parse_ps1_queue("﻿" + text) == P.queue_items()


# ---------------------------------------------------------------- 망가뜨린 입력 → FAIL

def _command_lines() -> list[int]:
    lines = ps1_lines()
    out = []
    for meta in P.STAGES.values():
        for n in line_range(meta["ps1_lines"]):
            s = lines[n - 1].strip()
            if s and not s.startswith("#"):
                out.append(n)
    return sorted(set(out))


COMMAND_LINES = _command_lines()


def test_command_line_inventory():
    assert len(COMMAND_LINES) == 26            # T1 4 + T2 2 + T3 1 + T4 2 + T5 7 + T6 4 + T7 2 + T8 4
    assert COMMAND_LINES == [20, 21, 22, 23, 25, 26, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                             41, 42, 43, 44, 47, 48, 49, 50, 51, 52]


@pytest.mark.parametrize("lineno", COMMAND_LINES)
def test_deleting_any_queue_line_breaks_parity(lineno):
    lines = ps1_lines()
    del lines[lineno - 1]
    broken = "\n".join(lines)
    assert P.queue_parity_errors(broken), f"ps1:{lineno}을 지웠는데 대조가 통과했다"
    try:
        got = P.parse_ps1_queue(broken)
    except ValueError:
        return                                  # 해석 실패도 FAIL로 드러난다
    assert got != P.queue_items()


def test_changed_foreach_values_break_parity_but_not_the_glob_set():
    text = ps1_text()
    assert "foreach ($s in 0, 1, 2, 3)" in text and "foreach ($s in 0, 1, 2)" in text
    fewer = text.replace("foreach ($s in 0, 1, 2, 3)", "foreach ($s in 0, 1, 2)")
    more = text.replace("foreach ($s in 0, 1, 2) {", "foreach ($s in 0, 1, 2, 3) {")
    for broken in (fewer, more):
        assert P.queue_parity_errors(broken)
        assert P.parse_ps1_queue(broken) != P.queue_items()
        # 집합 방식은 $s를 실재 파일 glob으로 펼치므로 이 변경을 보지 못한다
        assert glob_set_queued(broken) == glob_set_queued(text)


def test_swapped_seed_loop_lines_break_order_only():
    lines = ps1_lines()
    assert "cifar100_seeds_s$s" in lines[42 - 1] and "ood_cifar10_seeds_s$s" in lines[43 - 1]
    lines[41], lines[42] = lines[42], lines[41]
    broken = "\n".join(lines)
    got = P.parse_ps1_queue(broken)
    assert set(got) == set(P.queue_items())                    # 집합은 같다
    assert glob_set_queued(broken) == glob_set_queued(ps1_text())
    assert got != P.queue_items()                              # 순서 대조만 잡는다
    assert P.queue_parity_errors(broken)


@pytest.mark.parametrize("old,new", [
    ("configs\\synthetic.yaml", "config\\synthetic.yaml"),                    # 경로 오타
    ("--config configs\\cifar100.yaml", "--confg configs\\cifar100.yaml"),    # 옵션 오타
    ("--config configs\\cifar100_dml_lr3.yaml --only `",
     "--config configs\\cifar100_dml_lr3.yaml --only --force `"),           # --only 뒤 이름 없음
    ("foreach ($s in 0, 1, 2) {", "foreach ($s in 0, 1, x) {"),               # foreach 값 해석 불가
    ("& $py -m src.train --config configs\\cifar100_openset.yaml",
     "& $python -m src.train --config configs\\cifar100_openset.yaml"),       # 다른 실행 형태
])
def test_garbled_lines_raise(old, new):
    text = ps1_text()
    assert old in text
    broken = text.replace(old, new, 1)
    with pytest.raises(ValueError):
        P.parse_ps1_queue(broken)
    errs = P.queue_parity_errors(broken)
    assert errs and errs[0].startswith("ps1 해석 실패")


def test_unbalanced_foreach_blocks_raise():
    text = ps1_text()
    lines = ps1_lines()
    assert lines[52 - 1].strip() == "}"
    with pytest.raises(ValueError, match="닫히지"):
        P.parse_ps1_queue("\n".join(lines[:51] + lines[52:]))  # 52행 '}' 삭제(주석 속 {softmax,arcface}는 블록이 아니다)
    lines = ps1_lines()
    lines.insert(39, "}")                                      # 39행(빈 줄) 뒤에 짝 없는 '}'
    with pytest.raises(ValueError, match="짝이 없는"):
        P.parse_ps1_queue("\n".join(lines))
    with pytest.raises(ValueError, match="config 경로"):        # foreach 머리를 지우면 $s가 남는다
        P.parse_ps1_queue(text.replace("foreach ($s in 0, 1, 2) {", "", 1))
