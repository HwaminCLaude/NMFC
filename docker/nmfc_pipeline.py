"""nmfc_pipeline — scripts/run_all.ps1의 이식판.

QUEUE·STAGES·STEPS의 ps1 줄 번호는 지금 ps1에서 잰 물리 줄이다. 이 모듈은 ps1의 큐를 순서까지 그대로 옮기고,
ps1이 하지 않던 두 가지를 더한다. 하나는 near_ood를 seed_robustness보다 먼저 돌리는 순서 수정이고,
다른 하나는 단계별 실패 정책(허용 exit·사전조건·사후조건·필수 여부)이다.

독립성
    다른 nmfc_ 모듈을 import하지 않는다. 최상위 import는 stdlib와 yaml뿐이다(torch는 자식 프로세스만 쓴다).
    subprocess 실행기도 자체로 둔다: [sys.executable, "-B", ...], encoding="utf-8", errors="replace",
    자식 env에 PYTHONUTF8=1·PYTHONDONTWRITEBYTECODE=1·PYTHONUNBUFFERED=1(ps1의 `python -u` 대체).
    출력은 줄 단위로 흘려 보여 주고(30분짜리 학습이 조용하지 않게), 실행마다 runs/pipeline_logs/<시각>/에 전부 남긴다.

큐 (QUEUE, T1~T8) — ps1 줄 번호는 실측값이다
    T1 :20-23 synthetic --only 9런 · T2 :25-26 cifar100 --only 4런 · T3 :28 · T4 :30-31 --only 4런
    T5 :32-38 config 7개(39행은 빈 줄)
    T6 :41-44 foreach s=0,1,2,3 → cifar100_seeds_s{s}(:42) → ood_cifar10_seeds_s{s}(:43)
    T7 :47-48 · T8 :49-52 foreach s=0,1,2 → cifar100_openset_seeds_s{s}(:50) → ..._noise_grid_s{s}(:51)
    QUEUE 항목의 ps1_lines는 그 명령의 물리 줄이다(foreach 본문은 42·43·50·51). 단계 범위는 STAGES에 둔다.
    27회 호출, 127런. docker/tests/test_queue_parity.py가 ps1을 직접 펼친 결과와 대조한다.
    실행할 때도 workdir의 ps1을 다시 해석해 QUEUE와 다르면 아무것도 하지 않고 exit 2로 끝낸다.

후속 단계 (STEPS) — 평평한 하위 단계 목록. 각 항목의 group이 S1~S4, id는 "S3.near_ood" 꼴이다
    S1 bank      eval_softmax_bank ×2        {0}   ckpt가 있을 때만(있으면 필수)
    S2 ext       embed_external 래퍼(resnet50)  {0}   --with-ext일 때만, 선택(실패해도 전체 판정에 넣지 않음)
    S3 aggregate retrieval → fewshot_verify → stats_ci → near_ood → seed_robustness
                 → calibration → openset_detection → noise_grid   각 {0}
                 near_ood는 seed_robustness 앞이다(seed_robustness가 near_ood.json을 읽는다).
                 near_ood: ckpt 15개 중 하나라도 있으면 실행(없는 런은 스크립트가 건너뛰고 병합), 사후조건 15행.
                 calibration: ckpt 5개가 모두 있어야 실행(가드 없이 죽는다, eval_calibration.py:73).
                 openset_detection: c100v_softmax_open ckpt가 있어야 실행.
    S4 audit     audit_operating_regime {0} → audit_theory_claims {0}('통과 16/16')
    사전조건이 안 맞아 건너뛴 단계는 실패가 아니라 skipped_reason으로 기록한다.

모드 (run_pipeline 인자)
    stages=None, exp=None  → reproduce: 학습 T1~T8 + 후속 S1~S4(S2는 with_ext일 때만). from/to로 구간을 자른다.
    stages=[...] 또는 exp  → train: 고른 학습만. then_aggregate면 뒤에 집계 5종만 붙인다.
                              (train 하위 명령에서 '전체 학습만'은 stages=list(STAGE_IDS) 또는 to_step="T8")
    then_aggregate=True    → 후속 단계는 AGGREGATE_IDS(retrieval→fewshot_verify→stats_ci→near_ood(ckpt)→seed_robustness)만.
    from_step/to_step      → "T1".."T8", "S1".."S4" 또는 하위 id("S3.near_ood"). 전역 순서 T1<…<T8<S1<…<S4로 자른다.
    exp는 큐에서 찾는다(find_exp). 큐 밖 실험(결과를 저장소에 담지 않은 syn_moons_apt_none 등,
    진단용 diag224_* 등)은 LookupError → exit 2.

학습 판정
    항목마다 실행 전후 results/<exp>/metrics.json의 mtime_ns를 비교한다. 바뀌었거나 새로 생겼으면 '이번에 학습'.
    사후조건: force면 기대 런 전부, 아니면 실행 전에 metrics.json이 없던 런 = 이번에 학습한 런(집합 동일).
    학습 산출물 지문(items[*].artifacts): 학습 항목이 끝나면 이번에 학습한 런마다 results/<exp>/metrics.json과
    embeddings.npz의 원시 바이트 sha256·size·mtime_ns를 기록한다(npz는 이번 항목에서 다시 쓰였는지 rewritten도).
    compare는 지금 파일이 이 지문과 같을 때만 '이번 학습의 산출물'로 본다(학습 뒤 되돌리기·덮어쓰기 탐지).
    S1~S4 평가기는 학습 런의 metrics.json·embeddings.npz를 다시 쓰지 않는다(scripts/ 실측: 쓰기는 집계 JSON,
    *_open_bank·ext_*·openset_detection·openset_noise_grid, results/<exp>/near_ood.npz뿐).

실패 정책
    학습 런은 서로 독립이라 하나가 실패해도 다음 학습 항목을 계속한다(ps1의 실측 동작과 같다).
    학습이 하나라도 실패하면 후속 단계(S*)는 실행하지 않고 '학습 실패' 사유로 기록한다.
    후속 단계는 앞 산출물에 의존하므로 필수 단계가 실패하면 그 자리에서 멈추고 나머지는 '앞선 필수 단계 실패'로 남긴다.
    선택 단계(S2) 실패는 기록만 한다. 종료코드: 0 통과, 1 필수 항목 실패, 2 전제조건 불충족(인자·큐 대조·계획 오류).

상태 JSON (runs/pipeline_status.json, 항목마다 원자적으로 다시 쓴다)
    기본 키: started, finished, args, items[{id, kind, cmd, exit, ok, duration_s, trained, post[{cond, ok}],
    skipped_reason}], trained, pass.  추가 키(소비자는 무시해도 된다): version, started_ns, workdir, python,
    queue_parity, log_dir, exit_code, export, interrupted, items[*].{group, name, required, allowed_exit, expected,
    cached, started, started_ns, finished, notes, log, output_tail, artifacts}.  건너뛴 항목은 ok=None이다.
    artifacts = {exp: {"metrics.json": {sha256, size, mtime_ns} | None,
                       "embeddings.npz": {sha256, size, mtime_ns, rewritten} | None}} (학습 항목만, 이번에 학습한 런만).
    post에는 허용 exit 판정('exit N ∈ [...]')과 학습 런 수 판정도 들어간다. ok = post 전부 참.
    out이 주어지면 끝에(실패·중단이어도) <out>/reproduce/에 상태와 이번 실행 로그(pipeline_logs/)를 복사한다.

S2 래퍼와 scripts/embed_external.py 검토 결과 (import해도 되는가)
    모듈 수준 코드(:8-30)는 import(json·sys·pathlib·numpy·torch·torch.nn·DataLoader·torchvision), sys.path.insert(:18),
    `from src.evaluate import ...`(:20, src/evaluate.py 모듈 수준은 import·함수 정의뿐),
    `from src.utils.misc import get_device`(:21, misc.py 모듈 수준은 import·PROJECT_ROOT 상수·함수 정의뿐),
    상수 ROOT·DATA, `device = get_device()`(:25, torch.cuda.is_available() 조회뿐), 변환 객체 IMAGENET_TF(:27-30)다.
    파일 쓰기·네트워크·데이터 로딩·가중치 다운로드는 모두 함수 안에 있고, 다섯 모델 실행은
    `if __name__ == "__main__":`(:119-126) 안에 있다. 따라서 import는 안전하다(테스트가 AST로 다시 단언한다).
    래퍼(이 파일을 `--embed-resnet50`으로 자식 실행, cwd=workdir)는 run("ext_resnet50_frozen", m_resnet50) 하나만 부른다.
    run()(:63-70)은 모든 예외를 삼키고 [SKIP]만 찍으므로, 래퍼가 embeddings.npz의 mtime 갱신으로 성공을 판정한다.
    save()(:51-60)는 추적 파일 ext_resnet50_frozen/metrics.json도 덮어쓴다(tolerances.yaml의 retrain.expected_changes에 있다).
    사전조건: data/raw/cifar-100-python/{train,test,meta}(cifar_test가 download=False이고, torchvision CIFAR100의
    _check_integrity는 test만 읽을 때도 train_list+test_list를 모두 md5 검사한다, cifar.py:129-134),
    $TORCH_HOME/hub/checkpoints/resnet50-11ad3fa6.pth(ResNet50_Weights.IMAGENET1K_V2). 여기서는 존재만 보고,
    md5 불일치 등은 run()의 [SKIP] → 래퍼 exit 1로 드러난다. 외부 4종(ViT·ConvNeXt·CLIP·DINOv2)은 부르지 않는다.

디버그용 단독 실행(정식 진입점은 nmfc_cli.py train/reproduce):
    python -B docker/nmfc_pipeline.py --dry-run [--stage T1 ...] [--exp NAME] [--then aggregate] ...
    python -B docker/nmfc_pipeline.py --workdir <사본> [...]   (원본 저장소에 쓰지 않도록 실제 실행은 --workdir 필수)
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
_SELF = Path(__file__).resolve()
PS1_REL = "scripts/run_all.ps1"
STATUS_REL = "runs/pipeline_status.json"
LOGS_REL = "runs/pipeline_logs"
TAIL_LINES = 40

STAGE_IDS = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")
STEP_GROUPS = ("S1", "S2", "S3", "S4")
PHASES = STAGE_IDS + STEP_GROUPS

# ---------------------------------------------------------------- 학습 큐 (ps1 :20-52)

STAGES: dict[str, dict] = {
    "T1": {"ps1_lines": "20-23", "desc": "합성 2D 데이터 9런"},
    "T2": {"ps1_lines": "25-26", "desc": "v1 기본 설정 + 기본 lr(1e-3) 기준선 3종 4런"},
    "T3": {"ps1_lines": "28", "desc": "CE(lr 정렬) 1런"},
    "T4": {"ps1_lines": "30-31", "desc": "DML 4런"},
    "T5": {"ps1_lines": "32-38", "desc": "변형·scratch·OOD·개방셋 config 7개 15런"},
    "T6": {"ps1_lines": "41-44", "desc": "시드 복제 24런", "foreach": {"var": "s", "values": ("0", "1", "2", "3")}},
    "T7": {"ps1_lines": "47-48", "desc": "라벨오염 그리드 시드 42 16런"},
    "T8": {"ps1_lines": "49-52", "desc": "라벨오염 시드 축 54런", "foreach": {"var": "s", "values": ("0", "1", "2")}},
}


def _q(stage: str, config: str, only, lines: str) -> dict:
    return {"stage": stage, "config": config, "only": list(only) if only else None, "ps1_lines": lines}


QUEUE: list[dict] = [
    _q("T1", "configs/synthetic.yaml",
       ["syn_moons_linear", "syn_circles_linear", "syn_swiss_linear",
        "syn_moons_softmax", "syn_circles_softmax", "syn_swiss_softmax",
        "syn_moons_nmfc", "syn_circles_nmfc", "syn_swiss_nmfc"], "20-23"),
    _q("T2", "configs/cifar100.yaml",
       ["cifar100_nmfc", "cifar100_triplet", "cifar100_proto", "cifar100_deepmfa"], "25-26"),
    _q("T3", "configs/cifar100_softmax_lr3.yaml", None, "28"),
    _q("T4", "configs/cifar100_dml_lr3.yaml",
       ["cifar100_arcface_lr3", "cifar100_normsoftmax_lr3", "cifar100_ms_lr3", "cifar100_proxyanchor_plr"], "30-31"),
    _q("T5", "configs/cifar100_variants.yaml", None, "32"),
    _q("T5", "configs/cifar100_variants_wave2.yaml", None, "33"),
    _q("T5", "configs/cifar100_scratch.yaml", None, "34"),
    _q("T5", "configs/cifar100_dml_scratch.yaml", ["c100s_proxyanchor"], "35"),
    _q("T5", "configs/ood_cifar10_softmax_lr3.yaml", None, "36"),
    _q("T5", "configs/ood_cifar10_variants.yaml", None, "37"),
    _q("T5", "configs/cifar100_openset.yaml", None, "38"),
]
for _s in (0, 1, 2, 3):                       # ps1:41 foreach ($s in 0, 1, 2, 3)
    QUEUE.append(_q("T6", f"configs/cifar100_seeds_s{_s}.yaml", None, "42"))
    QUEUE.append(_q("T6", f"configs/ood_cifar10_seeds_s{_s}.yaml", None, "43"))
QUEUE += [
    _q("T7", "configs/cifar100_openset_noise.yaml", None, "47"),
    _q("T7", "configs/cifar100_openset_noise_grid.yaml", None, "48"),
]
for _s in (0, 1, 2):                          # ps1:49 foreach ($s in 0, 1, 2)
    QUEUE.append(_q("T8", f"configs/cifar100_openset_seeds_s{_s}.yaml", None, "50"))
    QUEUE.append(_q("T8", f"configs/cifar100_openset_noise_grid_s{_s}.yaml", None, "51"))
del _s


def queue_items() -> list[tuple[str, tuple[str, ...] | None]]:
    """QUEUE를 parse_ps1_queue와 같은 형식 [(config POSIX 경로, only 튜플|None), ...]으로."""
    return [(q["config"], tuple(q["only"]) if q["only"] else None) for q in QUEUE]


# ---------------------------------------------------------------- ps1 해석기

_TOKEN_RE = re.compile(r'"[^"]*"|\'[^\']*\'|\S+')
_FOREACH_RE = re.compile(r"^foreach\s*\(\s*\$(?P<var>[A-Za-z_]\w*)\s+in\s+(?P<vals>[^()]*?)\s*\)\s*\{$",
                         re.IGNORECASE)
_NAME_RE = re.compile(r"[A-Za-z0-9_]+")
_CONFIG_RE = re.compile(r"configs/[A-Za-z0-9_.\-]+\.yaml")
_QUOTED_RE = re.compile(r'"[^"]*"|\'[^\']*\'')


def _strip_comment(line: str) -> str:
    """따옴표 밖에서, 줄 처음이나 공백 뒤에 오는 #부터 줄 끝까지를 지운다."""
    quote = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1].isspace()):
            return line[:i]
    return line


def _logical_lines(text: str) -> list[tuple[int, int, str]]:
    """CRLF→LF 정규화 → 주석 제거 → 줄 끝 백틱(`) 연결. [(시작 줄, 끝 줄, 코드)], 빈 줄은 뺀다."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[tuple[int, int, str]] = []
    buf: list[str] = []
    start = None
    for no, raw in enumerate(lines, 1):
        code = _strip_comment(raw.lstrip("﻿") if no == 1 else raw)
        if "<#" in code:
            raise ValueError(f"ps1:{no}: 블록 주석(<# #>)은 해석하지 않는다")
        if start is None:
            start = no
        if code.endswith("`"):
            buf.append(code[:-1])
            continue
        buf.append(code)
        joined = " ".join(part.strip() for part in buf).strip()
        if joined:
            out.append((start, no, joined))
        buf, start = [], None
    if buf:
        raise ValueError("ps1 마지막 줄이 줄 연결 백틱(`)으로 끝난다")
    return out


def _foreach_values(spec: str, where: str) -> tuple[str, ...]:
    vals: list[str] = []
    for part in spec.split(","):
        p = part.strip()
        m = re.fullmatch(r"(-?\d+)\s*\.\.\s*(-?\d+)", p)
        if m:
            a, b = int(m[1]), int(m[2])
            step = 1 if b >= a else -1
            vals += [str(v) for v in range(a, b + step, step)]
        elif re.fullmatch(r"-?\d+", p):
            vals.append(str(int(p)))
        else:
            raise ValueError(f"ps1:{where}: foreach 값 {p!r}을 해석하지 못함")
    if not vals:
        raise ValueError(f"ps1:{where}: foreach 값이 비었다")
    return tuple(vals)


def _span(start: int, end: int) -> str:
    return f"{start}" if start == end else f"{start}-{end}"


def _parse_train_args(args: list[str], where: str) -> tuple[str, tuple[str, ...] | None]:
    cfg, only, i = None, None, 0
    while i < len(args):
        a = args[i]
        if a == "--config":
            if cfg is not None or i + 1 >= len(args):
                raise ValueError(f"ps1:{where}: --config 인자가 없거나 두 번 나온다")
            cfg, i = args[i + 1], i + 2
        elif a == "--only":
            if only is not None:
                raise ValueError(f"ps1:{where}: --only가 두 번 나온다")
            i += 1
            names = []
            while i < len(args) and not args[i].startswith("-"):
                names.append(args[i])
                i += 1
            if not names:
                raise ValueError(f"ps1:{where}: --only 뒤에 실험명이 없다")
            only = tuple(names)
        else:
            raise ValueError(f"ps1:{where}: 예상하지 못한 학습 인자 {a!r}")
    if cfg is None:
        raise ValueError(f"ps1:{where}: --config가 없다")
    cfg = cfg.replace("\\", "/")
    if "$" in cfg or not _CONFIG_RE.fullmatch(cfg):
        raise ValueError(f"ps1:{where}: config 경로를 해석하지 못함 {cfg!r}")
    bad = [n for n in (only or ()) if not _NAME_RE.fullmatch(n)]
    if bad:
        raise ValueError(f"ps1:{where}: 실험명 형식 오류 {bad}")
    return cfg, only


def _parse_command(code: str, start: int, end: int, subst: dict[str, str]) -> dict | None:
    """`& $py ...` 한 줄을 해석한다. 학습이면 kind=train, 스크립트면 kind=script, 그 밖은 None."""
    where = _span(start, end)
    toks: list[str] = []
    for t in _TOKEN_RE.findall(code):
        quote = t[0] if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'" else None
        val = t[1:-1] if quote else t
        if quote != "'":                                   # 작은따옴표 안은 PowerShell도 보간하지 않는다
            for var, v in subst.items():
                val = re.sub(r"\$\{" + re.escape(var) + r"\}|\$" + re.escape(var) + r"(?!\w)", v, val)
        toks.append(val)
    trainish = "src.train" in code or "--config" in code
    if len(toks) >= 2 and toks[0] == "&" and toks[1] == "$py":
        args = toks[2:]
        if args[:2] == ["-m", "src.train"]:
            cfg, only = _parse_train_args(args[2:], where)
            return {"kind": "train", "config": cfg, "only": only, "lines": where}
        if trainish or not args or args[0].startswith("-"):
            raise ValueError(f"ps1:{where}: 해석하지 못한 명령 줄: {code}")
        return {"kind": "script", "script": args[0].replace("\\", "/"), "args": tuple(args[1:]), "lines": where}
    if trainish:
        raise ValueError(f"ps1:{where}: 해석하지 못한 학습 줄: {code}")
    return None


def _parse_ps1(text: str) -> list[dict]:
    """ps1 전체를 순서대로 해석한다. foreach는 적힌 값 그대로 반복마다 본문을 펼친다(glob 아님)."""
    entries: list[dict] = []
    block = None
    for start, end, code in _logical_lines(text):
        m = _FOREACH_RE.match(code)
        if m:
            if block is not None:
                raise ValueError(f"ps1:{start}: 중첩 foreach는 지원하지 않는다")
            block = {"var": m["var"], "values": _foreach_values(m["vals"], str(start)),
                     "start": start, "body": []}
            continue
        if code == "}":
            if block is None:
                raise ValueError(f"ps1:{start}: 짝이 없는 '}}'")
            span = _span(block["start"], end)
            for v in block["values"]:
                for bs, be, bcode in block["body"]:
                    ent = _parse_command(bcode, bs, be, {block["var"]: v})
                    if ent is not None:
                        ent["foreach"] = {"var": block["var"], "value": v, "lines": span}
                        entries.append(ent)
            block = None
            continue
        bare = _QUOTED_RE.sub("", code)
        if "{" in bare or "}" in bare:
            raise ValueError(f"ps1:{_span(start, end)}: foreach 밖의 블록 구문은 해석하지 않는다: {code}")
        if block is not None:
            block["body"].append((start, end, code))
            continue
        ent = _parse_command(code, start, end, {})
        if ent is not None:
            entries.append(ent)
    if block is not None:
        raise ValueError(f"ps1:{block['start']}: foreach 블록이 닫히지 않았다")
    return entries


def parse_ps1_queue(text: str) -> list[tuple[str, tuple[str, ...] | None]]:
    """foreach 값을 그대로 펼친 (config POSIX 경로, only) 학습 호출 목록(ps1 순서). 해석 불가 줄은 ValueError."""
    return [(e["config"], e["only"]) for e in _parse_ps1(text) if e["kind"] == "train"]


def parse_ps1_scripts(text: str) -> list[tuple[str, tuple[str, ...]]]:
    """학습 뒤의 `& $py scripts\\x.py args` 호출 목록(ps1 순서). STEPS 대조용."""
    return [(e["script"], e["args"]) for e in _parse_ps1(text) if e["kind"] == "script"]


def _fmt_q(item) -> str:
    cfg, only = item
    return cfg + (f" --only {' '.join(only)}" if only else "")


def queue_parity_errors(ps1_text: str) -> list[str]:
    """ps1에서 펼친 큐와 QUEUE의 차이. 같으면 []."""
    try:
        got = parse_ps1_queue(ps1_text)
    except ValueError as e:
        return [f"ps1 해석 실패: {e}"]
    want = queue_items()
    if got == want:
        return []
    errs: list[str] = []
    if len(got) != len(want):
        errs.append(f"ps1 학습 호출 {len(got)}회 ≠ QUEUE {len(want)}항목")
    for i, (g, w) in enumerate(zip(got, want)):
        if g != w:
            errs.append(f"#{i + 1}번째부터 다름: ps1={_fmt_q(g)} / QUEUE={_fmt_q(w)}")
            break
    else:
        longer, name = (got, "ps1") if len(got) > len(want) else (want, "QUEUE")
        errs.append(f"{name}에만 있는 꼬리: {_fmt_q(longer[min(len(got), len(want))])}")
    return errs


# ---------------------------------------------------------------- config 해석 (src/train.py와 같은 규칙)

def _load_config(root: Path, config: str) -> dict:
    """train.load_config(:73-79)와 같다: base:가 있으면 한 단계 얕은 병합 {**base, **cfg}."""
    path = Path(root) / config
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"{config}: YAML 최상위가 사전이 아니다")
    if "base" in cfg:
        base = yaml.safe_load((path.parent / cfg["base"]).read_text(encoding="utf-8")) or {}
        cfg = {**base, **cfg}
    return cfg


def experiments_in(config: str, only, root: Path) -> list[str]:
    """config가 정의하는 실험 이름(config 순서 = 실제 학습 순서). only가 있으면 그중 only에 든 것만.

    train.run_from_config(:832-842)와 같은 해석이다: spec = {**defaults, **exp}, 이름은 spec["name"].
    only가 비어 있으면(None·빈 목록) train.py처럼 전부다. only에 정의되지 않은 이름이 있으면 ValueError —
    train.py는 그런 이름을 조용히 건너뛰므로(학습 0건) 여기서 막는다.
    """
    cfg = _load_config(Path(root), config)
    defaults = cfg.get("defaults") or {}
    names = [{**defaults, **exp}["name"] for exp in (cfg.get("experiments") or [])]
    if not only:
        return names
    unknown = [n for n in only if n not in names]
    if unknown:
        raise ValueError(f"{config}에 정의되지 않은 --only 이름: {unknown}")
    sel = set(only)
    return [n for n in names if n in sel]


def find_exp(name: str, root: Path) -> tuple[str, str]:
    """실험명 → (stage, config). 학습 큐(QUEUE)에서 찾는다. 큐 밖이면 LookupError."""
    root = Path(root)
    for q in QUEUE:
        if name in experiments_in(q["config"], q["only"], root):
            return q["stage"], q["config"]
    defined = []
    for p in sorted((root / "configs").glob("*.yaml")):
        rel = f"configs/{p.name}"
        try:
            if name in experiments_in(rel, None, root):
                defined.append(rel)
        except (ValueError, OSError, yaml.YAMLError):
            continue
    if defined:
        raise LookupError(f"{name}: {defined[0]}에 정의되어 있으나 학습 큐(run_all.ps1 T1~T8)에 없는 "
                          "런이다(결과를 저장소에 담지 않은 런 또는 진단용 런). 이 도구로는 학습하지 않는다")
    raise LookupError(f"{name}: configs/*.yaml 어디에도 정의되지 않은 실험이다")


# ---------------------------------------------------------------- 후속 단계 (ps1 :55-73)

NEAR_OOD_EXPS = tuple(b if s == 42 else f"{b}_s{s}"
                      for b in ("ood10v_shrb", "ood10v_mhrb", "ood_cifar10_softmax_lr3")
                      for s in (42, 0, 1, 2, 3))                      # eval_near_ood.py:36-37 BASES × SEEDS
CALIBRATION_CKPT_EXPS = ("cifar100_softmax_lr3", "cifar100_arcface_lr3", "cifar100_normsoftmax_lr3",
                         "cifar100_proxyanchor_plr", "cifar100_deepmfa")  # npz가 있는 비-NMFC 헤드 런
EXT_NPZ = "results/ext_resnet50_frozen/embeddings.npz"
CIFAR100_DIR = "data/raw/cifar-100-python"
RESNET50_WEIGHTS = "$TORCH_HOME/hub/checkpoints/resnet50-11ad3fa6.pth"


def _ckpt(exp: str) -> str:
    return f"checkpoints/{exp}/ckpt.pt"


def _fresh(rel: str) -> dict:
    return {"fresh": rel}


def _expect(s: str, n: int = 1) -> dict:
    return {"expect": s, "min": n}


def _step(id_: str, name: str, cmd: list[str], *, ps1: str | None, allowed=(0,), required: bool = True,
          needs=(), needs_any=(), post=(), aggregate: bool = False, with_ext_only: bool = False) -> dict:
    return {"id": id_, "group": id_.split(".", 1)[0], "name": name, "cmd": list(cmd),
            "allowed_exit": list(allowed), "required": required, "needs": list(needs),
            "needs_any": list(needs_any), "post": list(post), "aggregate": aggregate,
            "with_ext_only": with_ext_only, "ps1": ps1}


STEPS: list[dict] = [
    _step("S1.bank_softmax", "개방셋 참조뱅크 이식 평가(c100v_softmax_open 인코더)",
          ["scripts/eval_softmax_bank.py"], ps1="55", needs=[_ckpt("c100v_softmax_open")],
          post=[_fresh("results/c100v_softmax_open_bank/metrics.json")]),
    _step("S1.bank_shrb", "개방셋 참조뱅크 이식 평가(c100v_shrb_open 인코더)",
          ["scripts/eval_softmax_bank.py", "c100v_shrb_open"], ps1="56", needs=[_ckpt("c100v_shrb_open")],
          post=[_fresh("results/c100v_shrb_open_bank/metrics.json")]),
    _step("S2.ext", "외부 임베딩 ResNet50 동결(embed_external 래퍼, resnet50만)",
          [str(_SELF), "--embed-resnet50"], ps1="59", required=False, with_ext_only=True,
          needs=[f"{CIFAR100_DIR}/train", f"{CIFAR100_DIR}/test", f"{CIFAR100_DIR}/meta", RESNET50_WEIGHTS],
          post=[_fresh(EXT_NPZ)]),
    _step("S3.retrieval", "검색 평가 → retrieval_eval.json", ["scripts/eval_retrieval.py"], ps1="62",
          aggregate=True, post=[_fresh("results/retrieval_eval.json")]),
    _step("S3.fewshot_verify", "few-shot·검증 → fewshot_verify_eval.json", ["scripts/eval_fewshot_verify.py"],
          ps1="63", aggregate=True, post=[_fresh("results/fewshot_verify_eval.json")]),
    _step("S3.stats_ci", "에피소드·부트스트랩 CI → stats_ci.json", ["scripts/eval_stats_ci.py"], ps1="64",
          aggregate=True, post=[_fresh("results/stats_ci.json")]),
    _step("S3.near_ood", "near-OOD 사후 채점 → near_ood.json (seed_robustness보다 먼저)",
          ["scripts/eval_near_ood.py"], ps1="66", aggregate=True, needs_any=[_ckpt(e) for e in NEAR_OOD_EXPS],
          post=[_fresh("results/near_ood.json"),
                {"rows": "results/near_ood.json", "n": len(NEAR_OOD_EXPS), "keys": list(NEAR_OOD_EXPS)}]),
    _step("S3.seed_robustness", "시드 강건성 → seed_robustness.json", ["scripts/eval_seed_robustness.py"],
          ps1="65", aggregate=True, post=[_fresh("results/seed_robustness.json")]),
    _step("S3.calibration", "보정 평가 → calibration_eval.json", ["scripts/eval_calibration.py"], ps1="67",
          needs=[_ckpt(e) for e in CALIBRATION_CKPT_EXPS], post=[_fresh("results/calibration_eval.json")]),
    _step("S3.openset_detection", "개방셋 탐지 → openset_detection/", ["scripts/eval_openset_detection.py"],
          ps1="68", needs=[_ckpt("c100v_softmax_open")],
          post=[_fresh("results/openset_detection/auroc_table.md")]),
    _step("S3.noise_grid", "라벨오염 그리드 → openset_noise_grid/", ["scripts/eval_openset_noise_grid.py"],
          ps1="69", post=[_fresh("results/openset_noise_grid/master.json")]),
    _step("S4.regime", "작동점 감사", ["scripts/audit_operating_regime.py"], ps1="72",
          post=[_fresh("results/operating_regime_audit.json")]),
    _step("S4.theory", "명제 검증 16/16", ["scripts/audit_theory_claims.py"], ps1="73",
          post=[_expect("통과 16/16"), _fresh("results/theory_audit.json")]),
]
STEP_BY_ID: dict[str, dict] = {s["id"]: s for s in STEPS}
AGGREGATE_IDS: tuple[str, ...] = tuple(s["id"] for s in STEPS if s["aggregate"])


def exit_allowed(step: dict, code: int | None) -> bool:
    """허용 exit 판정. step은 STEPS 항목(또는 allowed_exit를 가진 사전)."""
    return code is not None and code in step["allowed_exit"]


# ---------------------------------------------------------------- 계획

def _norm_id(pid: str) -> str:
    pid = pid.strip()
    return pid[:2].upper() + pid[2:] if len(pid) >= 2 else pid.upper()


def _phase_key(pid: str) -> tuple[int, int | None]:
    pid = _norm_id(pid)
    if pid in PHASES:
        return PHASES.index(pid), None
    if pid in STEP_BY_ID:
        g = STEP_BY_ID[pid]["group"]
        subs = [s["id"] for s in STEPS if s["group"] == g]
        return PHASES.index(g), subs.index(pid)
    raise ValueError(f"알 수 없는 단계 ID {pid!r} — T1~T8, S1~S4 또는 하위 단계({', '.join(STEP_BY_ID)})")


def _train_item(id_: str, stage: str, config: str, only, root: Path, force: bool, sub: int,
                ps1_lines: str | None) -> dict:
    cmd = ["-m", "src.train", "--config", config]
    if only:
        cmd += ["--only", *only]
    if force:
        cmd.append("--force")
    return {"id": id_, "kind": "train", "group": stage, "sub": sub, "name": f"학습 {config}",
            "cmd": cmd, "config": config, "only": list(only) if only else None,
            "expected": experiments_in(config, only, root), "force": force,
            "allowed_exit": [0], "required": True, "ps1_lines": ps1_lines, "static_skip": None}


def _step_item(step: dict, with_ext: bool) -> dict:
    subs = [s["id"] for s in STEPS if s["group"] == step["group"]]
    it = {"id": step["id"], "kind": "step", "group": step["group"], "sub": subs.index(step["id"]),
          "name": step["name"], "cmd": list(step["cmd"]), "allowed_exit": list(step["allowed_exit"]),
          "required": step["required"], "step": step, "static_skip": None}
    if step["with_ext_only"] and not with_ext:
        it["static_skip"] = "--with-ext 미지정(외부 임베딩은 선택 단계)"
    return it


def build_plan(root: Path, *, stages: list[str] | None = None, exp: str | None = None, force: bool = False,
               with_ext: bool = False, from_step: str | None = None, to_step: str | None = None,
               then_aggregate: bool = False) -> list[dict]:
    """실행할 항목을 순서대로 만든다(파일을 쓰지 않는다). 인자 오류는 ValueError, 큐 밖 --exp는 LookupError."""
    root = Path(root)
    if isinstance(stages, str):
        stages = [stages]
    if stages:
        stages = [_norm_id(x) for s in stages for x in str(s).split(",") if x.strip()]
        bad = [s for s in stages if s not in STAGE_IDS]
        if bad:
            raise ValueError(f"알 수 없는 학습 단계 {bad} (T1~T8)")
    if exp is not None and stages:
        raise ValueError("--stage와 --exp는 함께 쓸 수 없다")
    lo = _phase_key(from_step) if from_step else None
    hi = _phase_key(to_step) if to_step else None

    items: list[dict] = []
    if exp is not None:
        stage, config = find_exp(exp, root)
        lines = next(q["ps1_lines"] for q in QUEUE if q["config"] == config)
        items.append(_train_item(f"{stage}:{exp}", stage, config, [exp], root, force, 0, lines))
    else:
        seen: collections.Counter = collections.Counter()
        for q in QUEUE:
            k = seen[q["stage"]]
            seen[q["stage"]] += 1
            if stages and q["stage"] not in stages:
                continue
            items.append(_train_item(f"{q['stage']}.{k + 1}", q["stage"], q["config"], q["only"], root,
                                     force, k, q["ps1_lines"]))
    train_mode = exp is not None or bool(stages)
    if then_aggregate:
        steps = [s for s in STEPS if s["aggregate"]]
    elif train_mode:
        steps = []
    else:
        steps = list(STEPS)
    items += [_step_item(s, with_ext) for s in steps]

    def key(it: dict) -> tuple[int, int]:
        return PHASES.index(it["group"]), it["sub"]

    if lo:
        items = [it for it in items if key(it) >= (lo[0], -1 if lo[1] is None else lo[1])]
    if hi:
        items = [it for it in items if key(it) <= (hi[0], 10 ** 9 if hi[1] is None else hi[1])]
    return items


# ---------------------------------------------------------------- 실행기

def child_env(extra: dict | None = None) -> dict:
    """os.environ + extra, 그 위에 UTF-8·바이트코드 금지·무버퍼를 강제한다(extra가 덮어쓰지 못한다).

    PYTHONIOENCODING도 강제한다: 사용자 환경에 PYTHONIOENCODING=cp949 등이 있으면 PYTHONUTF8=1이어도 자식의
    표준 입출력 인코딩이 그것을 따르므로, UTF-8로 읽는 부모가 기대 문자열을 놓친다(nmfc_common.child_env와 같은 규칙).
    """
    env = dict(os.environ)
    if extra:
        env.update(extra)
    env.update({"PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
                "PYTHONIOENCODING": "utf-8"})
    return env


def _emit(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(msg.encode(enc, "replace").decode(enc, "replace"), flush=True)


def _now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="milliseconds")


def _mtime(p: Path) -> int | None:
    try:
        return p.stat().st_mtime_ns
    except OSError:
        return None


ARTIFACT_FILES = ("metrics.json", "embeddings.npz")


def file_fingerprint(p: Path) -> dict | None:
    """원시 바이트 sha256·size·mtime_ns. 파일이 없으면 None(compare와 같은 정의: 정규화하지 않는다)."""
    try:
        st = p.stat()
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return {"sha256": h.hexdigest(), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def run_artifacts(root: Path, exp: str, npz_before: int | None) -> dict:
    """학습 직후 results/<exp>/의 metrics.json·embeddings.npz 지문. npz에는 이번 항목에서 다시 쓰였는지(rewritten)."""
    d = Path(root) / "results" / exp
    out = {name: file_fingerprint(d / name) for name in ARTIFACT_FILES}
    if out["embeddings.npz"] is not None:
        out["embeddings.npz"]["rewritten"] = out["embeddings.npz"]["mtime_ns"] != npz_before
    return out


def _torch_home() -> Path:
    """torch.hub._get_torch_home과 같은 규칙(torch를 import하지 않으려고 옮겨 적음)."""
    th = os.environ.get("TORCH_HOME")
    if th:
        return Path(os.path.expanduser(th))
    return Path(os.path.expanduser(os.path.join(os.environ.get("XDG_CACHE_HOME", "~/.cache"), "torch")))


def _need_path(root: Path, rel: str) -> Path:
    if rel.startswith("$TORCH_HOME/"):
        return _torch_home() / rel[len("$TORCH_HOME/"):]
    return root / rel


def _check_needs(step: dict, root: Path) -> str | None:
    """사전조건을 본다. 안 맞으면 건너뛸 사유 문자열, 맞으면 None."""
    missing = [r for r in step["needs"] if not _need_path(root, r).exists()]
    if missing:                                    # needs는 많아야 5개라 전부 보여 준다
        return f"사전조건 미충족 — 없음: {', '.join(missing)}"
    if step["needs_any"] and not any(_need_path(root, r).exists() for r in step["needs_any"]):
        return f"사전조건 미충족 — {len(step['needs_any'])}개 중 하나도 없음(예: {step['needs_any'][0]})"
    return None


def _resolve_step(step: dict, root: Path) -> tuple[list[str], list[dict], list[str]]:
    """실행 시점의 최종 명령·사후조건·메모. 모든 단계가 정적이므로 STEPS 값을 복사해 돌려준다."""
    return list(step["cmd"]), list(step["post"]), []


def _display(cmd: list[str]) -> str:
    return " ".join(f'"{c}"' if (not c or any(ch.isspace() for ch in c)) else c for c in cmd)


def _run_cmd(cmd: list[str], cwd: Path, env: dict, log_path: Path,
             expect: dict[str, int] | None = None) -> tuple[int, list[str], dict[str, int], float]:
    """출력을 줄 단위로 흘려 보이고 로그에 쓰며 마지막 TAIL_LINES줄과 기대 문자열 출현 수를 센다."""
    tail: collections.deque = collections.deque(maxlen=TAIL_LINES)
    counts = {k: 0 for k in (expect or {})}
    t0 = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8", newline="\n") as log:
        log.write(f"$ {_display(cmd)}\n# cwd={cwd}\n")
        try:
            proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    encoding="utf-8", errors="replace", bufsize=1)
        except OSError as e:
            msg = f"[pipeline] 실행 실패: {type(e).__name__}: {e}"
            log.write(msg + "\n")
            _emit(msg)
            return 127, [msg], counts, time.monotonic() - t0
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip("\n")
                log.write(line + "\n")
                tail.append(line)
                for k in counts:
                    counts[k] += line.count(k)
                _emit(line)
            rc = proc.wait()
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        log.write(f"# exit={rc}\n")
    return rc, list(tail), counts, time.monotonic() - t0


def _eval_post(post: list[dict], root: Path, before: dict[str, int | None], counts: dict[str, int]) -> list[dict]:
    res = []
    for p in post:
        if "fresh" in p:
            rel = p["fresh"]
            now = _mtime(root / rel)
            ok = now is not None and now != before.get(rel)
            res.append({"cond": f"갱신됨: {rel}", "ok": ok,
                        "detail": "파일 없음" if now is None else ("mtime 그대로" if not ok else "mtime 변경")})
        elif "rows" in p:
            rel, n = p["rows"], p["n"]
            try:
                data = json.loads((root / rel).read_text(encoding="utf-8"))
                keys = set(data) if isinstance(data, dict) else set()
                lack = sorted(set(p.get("keys") or []) - keys)
                ok = isinstance(data, dict) and len(data) == n and not lack
                detail = f"{len(data)}행" + (f", 빠진 키 {lack}" if lack else "")
            except (OSError, ValueError) as e:
                ok, detail = False, f"읽기 실패: {type(e).__name__}"
            res.append({"cond": f"{rel} {n}행", "ok": ok, "detail": detail})
        elif "expect" in p:
            got = counts.get(p["expect"], 0)
            res.append({"cond": f"출력에 '{p['expect']}' {p['min']}회 이상", "ok": got >= p["min"],
                        "detail": f"{got}회"})
    return res


def _write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:                     # Windows: 다른 프로세스가 잠깐 열고 있을 때
            if attempt == 4:
                raise
            time.sleep(0.2)


def _new_record(it: dict) -> dict:
    return {"id": it["id"], "kind": it["kind"], "group": it["group"], "name": it["name"],
            "cmd": [sys.executable, "-B", *it["cmd"]], "required": it["required"],
            "allowed_exit": list(it["allowed_exit"]), "exit": None, "ok": None, "duration_s": None,
            "started": None, "started_ns": None, "finished": None,
            "trained": [], "expected": it.get("expected"), "cached": [], "artifacts": {},
            "post": [], "skipped_reason": None, "notes": [], "log": None, "output_tail": []}


def _log_rel(log_dir: str, idx: int, item_id: str) -> str:
    return f"{log_dir}/{idx:03d}_{re.sub(r'[^A-Za-z0-9_.-]', '_', item_id)}.log"


def _run_item(it: dict, rec: dict, root: Path, env: dict, idx: int, log_dir: str) -> None:
    if it["kind"] == "train":
        cmd_tail, post, notes = it["cmd"], [], []
        metrics = {e: root / "results" / e / "metrics.json" for e in it["expected"]}
        before_m = {e: _mtime(p) for e, p in metrics.items()}
        before_npz = {e: _mtime(root / "results" / e / "embeddings.npz") for e in it["expected"]}
        to_train = list(it["expected"]) if it["force"] else [e for e in it["expected"] if before_m[e] is None]
        rec["cached"] = [e for e in it["expected"] if e not in to_train]
        before: dict[str, int | None] = {}
    else:
        cmd_tail, post, notes = _resolve_step(it["step"], root)
        before = {p["fresh"]: _mtime(root / p["fresh"]) for p in post if "fresh" in p}
    cmd = [sys.executable, "-B", *cmd_tail]
    expect = {p["expect"]: p["min"] for p in post if "expect" in p}
    rec.update(cmd=cmd, notes=notes, log=_log_rel(log_dir, idx, it["id"]), started=_now_iso(),
               started_ns=time.time_ns())
    _emit(f"[pipeline] >> {it['id']} ({it['name']}): python -B {_display(cmd_tail)}")
    for n in notes:
        _emit(f"[pipeline]    {n}")
    rc, tail, counts, dur = _run_cmd(cmd, root, env, root / rec["log"], expect)
    rec.update(exit=rc, duration_s=round(dur, 3), finished=_now_iso(), output_tail=tail)
    checks = [{"cond": f"exit {rc} ∈ {sorted(it['allowed_exit'])}", "ok": rc in it["allowed_exit"]}]
    if it["kind"] == "train":
        rec["trained"] = [e for e, p in metrics.items()
                          if _mtime(p) is not None and _mtime(p) != before_m[e]]
        rec["artifacts"] = {e: run_artifacts(root, e, before_npz.get(e)) for e in rec["trained"]}
        how = "--force: 기대 런 전부" if it["force"] else "캐시(metrics.json) 없던 런"
        checks.append({"cond": f"재학습 {len(rec['trained'])}런 = 기대 {len(to_train)}런({how})",
                       "ok": set(rec["trained"]) == set(to_train),
                       "detail": f"기대 {to_train}, 실제 {rec['trained']}"})
    checks += _eval_post(post, root, before, counts)
    rec["post"] = checks
    rec["ok"] = all(c["ok"] for c in checks)
    bad = [c["cond"] for c in checks if not c["ok"]]
    _emit(f"[pipeline] {'ok  ' if rec['ok'] else 'FAIL'} {it['id']} exit={rc} {dur:.1f}s"
          + (f" 재학습 {len(rec['trained'])}런" if it["kind"] == "train" else "")
          + (f" — 불충족: {bad}" if bad else ""))


def _export(root: Path, out: Path, status: dict, status_path: Path) -> None:
    dest = Path(out) / "reproduce"
    dest.mkdir(parents=True, exist_ok=True)
    done = []
    logs = root / status.get("log_dir", LOGS_REL)
    if logs.is_dir():                                   # 이번 실행의 로그만(지난 실행 로그와 섞지 않는다)
        shutil.rmtree(dest / "pipeline_logs", ignore_errors=True)
        shutil.copytree(logs, dest / "pipeline_logs")
        done.append("pipeline_logs/")
    status["export"] = {"dir": str(dest), "files": ["pipeline_status.json", *done]}
    _write_json_atomic(status_path, status)
    shutil.copy2(status_path, dest / "pipeline_status.json")


def _queue_parity_at(root: Path) -> dict:
    p = root / PS1_REL
    if not p.is_file():
        return {"ok": None, "errors": [f"{PS1_REL} 없음 — 큐 대조 생략"]}
    errs = queue_parity_errors(p.read_text(encoding="utf-8-sig"))
    return {"ok": not errs, "errors": errs}


def _print_plan(plan: list[dict], root: Path, parity: dict) -> None:
    _emit(f"[dry-run] workdir={root}")
    _emit(f"[dry-run] python={sys.executable} -B (자식 env: PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 "
          "PYTHONUNBUFFERED=1)")
    _emit("[dry-run] 큐 대조(ps1): " + ("OK" if parity["ok"] else "; ".join(parity["errors"])))
    n_runs = 0
    for i, it in enumerate(plan, 1):
        notes: list[str] = []
        if it["kind"] == "train":
            state, cmd = "run", it["cmd"]
            n_runs += len(it["expected"])
            notes.append(f"기대 런 {len(it['expected'])}개 (ps1:{it['ps1_lines']})")
        elif it["static_skip"]:
            state, cmd = "skip", it["cmd"]
            notes.append(it["static_skip"])
        else:
            step = it["step"]
            cmd, _post, extra = _resolve_step(step, root)
            reason = _check_needs(step, root)
            state = "skip" if reason else "run"
            notes += ([reason] if reason else []) + extra
            if step["needs"] or step["needs_any"]:
                notes.append("조건은 실행 시점에 다시 판정한다")
        _emit(f"[dry-run] {i:3d} {it['id']:<22} {state:<4} python -B {_display(cmd)}")
        for n in notes:
            _emit(f"[dry-run]       -> {n}")
    _emit(f"[dry-run] 합계: 항목 {len(plan)}개, 학습 런 {n_runs}개")


def run_pipeline(workdir: Path, *, stages: list[str] | None = None, exp: str | None = None, force: bool = False,
                 with_ext: bool = False, from_step: str | None = None, to_step: str | None = None,
                 then_aggregate: bool = False, out: Path | None = None, dry_run: bool = False) -> int:
    """파이프라인을 실행하고 runs/pipeline_status.json에 기록한다. dry_run이면 계획만 출력하고 아무것도 쓰지 않는다.

    반환: 0 통과, 1 필수 항목 실패, 2 전제조건 불충족(인자 오류·큐 밖 --exp·ps1 큐 불일치·빈 계획).
    """
    root = Path(workdir).resolve()
    args = {"stages": list(stages) if stages else None, "exp": exp, "force": force, "with_ext": with_ext,
            "from_step": from_step, "to_step": to_step, "then_aggregate": then_aggregate,
            "out": str(out) if out else None, "dry_run": dry_run}
    try:
        plan = build_plan(root, stages=stages, exp=exp, force=force, with_ext=with_ext, from_step=from_step,
                          to_step=to_step, then_aggregate=then_aggregate)
    except (ValueError, LookupError, OSError, yaml.YAMLError) as e:
        _emit(f"[pipeline] 계획 오류: {e}")
        return 2
    if not plan:
        _emit("[pipeline] 실행할 항목이 없다(--stage/--exp/--from/--to 조합 확인)")
        return 2
    parity = _queue_parity_at(root)
    if dry_run:
        _print_plan(plan, root, parity)
        return 2 if parity["ok"] is False else 0

    status_path = root / STATUS_REL
    log_dir = f"{LOGS_REL}/{_dt.datetime.now():%Y%m%d-%H%M%S-%f}"
    status = {"version": 1, "started": _now_iso(), "started_ns": time.time_ns(), "finished": None,
              "args": args, "workdir": str(root), "python": sys.executable, "queue_parity": parity,
              "log_dir": log_dir, "items": [], "trained": [], "pass": None, "exit_code": None}
    if parity["ok"] is False:
        _emit("[pipeline] ps1 큐와 QUEUE가 다르다 — 실행하지 않는다: " + "; ".join(parity["errors"]))
        status.update(finished=_now_iso(), **{"pass": False, "exit_code": 2})
        _write_json_atomic(status_path, status)
        if out:
            _export(root, out, status, status_path)
        return 2

    env = child_env()
    train_failed, stop_reason, exit_code = False, None, 1
    try:
        _write_json_atomic(status_path, status)
        for idx, it in enumerate(plan, 1):
            rec = _new_record(it)
            status["items"].append(rec)
            if stop_reason:
                rec["skipped_reason"] = stop_reason
            elif it["kind"] == "step" and train_failed:
                rec["skipped_reason"] = "학습 실패 — 후속 단계를 실행하지 않음"
            elif it["static_skip"]:
                rec["skipped_reason"] = it["static_skip"]
            elif it["kind"] == "step" and (reason := _check_needs(it["step"], root)):
                rec["skipped_reason"] = reason
            else:
                _run_item(it, rec, root, env, idx, log_dir)
                status["trained"] += rec["trained"]
                if not rec["ok"] and it["required"]:
                    if it["kind"] == "train":
                        train_failed = True
                    else:
                        stop_reason = f"앞선 필수 단계 {it['id']} 실패로 중단"
                elif not rec["ok"]:
                    _emit(f"[pipeline] 선택 단계 {it['id']} 실패 — 전체 판정에는 넣지 않는다")
            if rec["skipped_reason"]:
                _emit(f"[pipeline] skip {it['id']}: {rec['skipped_reason']}")
            _write_json_atomic(status_path, status)
        passed = not any(r["required"] and r["exit"] is not None and not r["ok"] for r in status["items"])
        exit_code = 0 if passed else 1
    except KeyboardInterrupt:
        status["interrupted"] = True
        exit_code = 130
        raise
    finally:
        status.update(finished=_now_iso(), exit_code=exit_code, **{"pass": exit_code == 0})
        _write_json_atomic(status_path, status)
        if out:
            try:
                _export(root, out, status, status_path)
            except OSError as e:
                _emit(f"[pipeline] --out 복사 실패: {e}")
    n_ok = sum(1 for r in status["items"] if r["ok"])
    n_skip = sum(1 for r in status["items"] if r["skipped_reason"])
    _emit(f"[pipeline] {'PASS' if exit_code == 0 else 'FAIL'} — 항목 {len(status['items'])}개(성공 {n_ok}, "
          f"건너뜀 {n_skip}), 이번에 학습 {len(status['trained'])}런 → {status_path}")
    return exit_code


# ---------------------------------------------------------------- S2 래퍼 본체 (자식 프로세스 전용)

def _embed_resnet50_main() -> int:
    """cwd(workdir)의 scripts/embed_external.py를 import해 resnet50 경로 하나만 실행한다. 판정은 npz mtime 갱신."""
    import importlib.util

    root = Path.cwd()
    script = root / "scripts" / "embed_external.py"
    npz = root / EXT_NPZ
    if not script.is_file():
        print(f"[FAIL] {script} 없음", flush=True)
        return 2
    before = _mtime(npz)
    spec = importlib.util.spec_from_file_location("embed_external", script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["embed_external"] = mod
    spec.loader.exec_module(mod)
    print(f"[device] {mod.device}", flush=True)
    mod.run("ext_resnet50_frozen", mod.m_resnet50)
    after = _mtime(npz)
    if after is None or after == before:
        print(f"[FAIL] {EXT_NPZ}가 갱신되지 않았다 — 위 [SKIP] 사유 참고"
              "(embed_external.run은 예외를 삼키고 exit 0으로 끝난다)", flush=True)
        return 1
    print("done (ext_resnet50_frozen만 실행; 외부 4종은 지원하지 않음)", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="nmfc_pipeline.py",
                                 description="run_all.ps1 이식판(디버그용; 정식 진입점은 nmfc_cli.py)")
    ap.add_argument("--embed-resnet50", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--workdir", default=None, help="실행할 저장소 사본(실제 실행에는 필수)")
    ap.add_argument("--stage", nargs="+", dest="stages", default=None)
    ap.add_argument("--exp", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--with-ext", action="store_true")
    ap.add_argument("--from", dest="from_step", default=None)
    ap.add_argument("--to", dest="to_step", default=None)
    ap.add_argument("--then", choices=["aggregate"], default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.embed_resnet50:
        return _embed_resnet50_main()
    if a.workdir is None and not a.dry_run:
        ap.error("실제 실행에는 --workdir가 필요하다(원본 저장소에 직접 쓰지 않기 위해). 미리보기는 --dry-run")
    return run_pipeline(Path(a.workdir) if a.workdir else REPO_ROOT, stages=a.stages, exp=a.exp, force=a.force,
                        with_ext=a.with_ext, from_step=a.from_step, to_step=a.to_step,
                        then_aggregate=a.then == "aggregate", out=Path(a.out) if a.out else None,
                        dry_run=a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
