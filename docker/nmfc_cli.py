"""nmfc_cli — NMFC 재현 도구 진입점.

사용
    호스트(Windows·Linux): python docker/nmfc_cli.py <명령> [옵션]
    컨테이너:              nmfc <명령> [옵션]   (/usr/local/bin/nmfc 래퍼)

하위 명령: env · verify · regen · fetch · smoke · train · reproduce · compare · manifest
공통 옵션: --out(보고서 위치, 기본 $NMFC_OUT 또는 <저장소>/out) · --workdir · --keep · --baseline
종료코드: 0 PASS · 1 FAIL · 2 전제조건 불충족(판정 불가) · 130 사용자 중단
보고서:   <out>/<명령>_report.{json,md} (fetch는 fetch_data_report.* / fetch_weights_report.*)
결과표:   <out>/<명령>/<표>.csv(utf-8-sig)·.md (docker/nmfc_tables.py) — verify: theory_claims, regen: aggregates·table_files,
          train: trained_runs, compare: compare_table·gates. 판정 열 값은 PASS·FAIL·REPORT·N/A.
          쓰기 실패는 WARN 검사(종료코드 불변).

이 모듈에서 정한 세부
 1. 디스패치
    - argv[0]이 'manifest'면 argparse를 거치지 않고 nmfc_manifest.run_manifest_cli(argv)에 그대로 넘긴다.
      공통 옵션 --baseline과 `manifest --baseline DEST`가 겹치지 않게 하려는 것이다. 이 경로와 --help는
      torch·matplotlib·numpy를 import하지 않는다(빌드 중 finalize.sh가 부른다).
    - '_'로 시작하는 숨은 하위 명령(_device-info, _smoke-child, _tv-integrity)은 이 CLI가 자식 프로세스로
      부르는 내부용이다. --help에 나오지 않는다.
 2. CLI 프로세스는 torch·src를 import하지 않는다. 장치 정보, smoke 텐서 검사, torchvision 무결성 검사는
    [sys.executable, -B, nmfc_cli.py, _숨은명령] 자식에서 돌린다. 그래서 장치 강제 env(CUDA_VISIBLE_DEVICES=-1)가
    그대로 적용되고, OOM·bus error가 CLI를 죽이지 않는다. matplotlib(한글 폰트 단언)와 nmfc_compare
    (numpy·PIL·scipy)는 하위 명령 안에서 지연 import한다.
 3. 작업 트리
    - 컨테이너: /workspace/nmfc에서 그대로 실행한다.
    - 호스트: 기본은 make_workdir의 일회용 사본(%TEMP%\\nmfc-work-<pid>, SKIP_DIRS 제외)이다.
      끝나면 지우고, --keep이면 남긴다.
    - --workdir DIR: 그 트리에서 그대로 실행하고 **절대 지우지 않는다**(사용자 경로).
      호스트에서 트리에 쓰는 명령(verify·regen·train·reproduce)은 DIR이 원본 저장소(REPO_ROOT)면 거부한다(exit 2).
      compare는 읽기만 하므로 원본 저장소도 허용한다. smoke는 DIR에서 src/·configs/만 읽는다.
 4. 기준본(--baseline)
    - 지정하면 그것을 쓴다. 컨테이너 기본은 $NMFC_BASELINE(/opt/nmfc/baseline).
    - 호스트 verify·regen: 작업 트리를 정한 직후(단계 실행 전) make_baseline으로 저장소 밖 임시 폴더
      (%TEMP%\\nmfc-baseline-<pid>-*)에 스냅샷을 뜬다. --keep이 아니면 끝나고 지운다.
    - 호스트 train·reproduce: 학습 전에 <workdir>/runs/nmfc_baseline에 스냅샷을 뜬다. runs/는 SKIP_DIRS라
      lock·사본·대조에 섞이지 않는다. 이미 있으면 그대로 둔다(가장 이른 상태 = 학습 전 상태를 유지).
    - 호스트 compare: --baseline → <workdir>/runs/nmfc_baseline → 원본 저장소 스냅샷 순으로 고른다.
 5. verify의 '예상 밖 변경 0건'
    - <workdir>/docker/context.lock이 있으면 verify_lock(ignore_extra_under=())로 **엄격하게** 대조한다
      (기본 무시 목록은 results/·logs/ 아래 새 파일을 숨기므로 쓰지 않는다).
    - lock이 없으면(지금의 호스트) 단계 실행 직전 list_files 전체의 LF 정규화 sha256을 메모리에 떠 두고 대조한다.
      작업 트리에 lock을 쓰지 않는다(--workdir가 빌드 컨텍스트일 때 진짜 lock을 덮어쓰지 않게).
    - expected.yaml writes 경로의 'changed'·'extra'만 허용한다. 그 밖의 changed·missing·extra는 FAIL이다.
 6. regen은 REGEN_ORDER(집계 평가기 10단계) 순서를 nmfc_pipeline.STEP_BY_ID로 만든다(ckpt 사전조건·사후조건을 그대로 재사용).
    ckpt가 없는 단계는 실패가 아니라 N/A(측정 불가) 검사로 남긴다. 장치는 자식 프로세스가 실제로 쓰는 장치를
    _device-info 자식으로 재서 regen_mode·regen_label에 넘긴다(--device cpu면 CUDA_VISIBLE_DEVICES=-1).
 7. smoke(①~⑥)는 src/·configs/만 담은 임시 사본(%TEMP%\\nmfc-smoke-*)에서 한다.
    ④ 로더는 src.datasets.image.make_train_loader('shuffle')·make_eval_loader를 그대로 쓰고, 데이터셋은 인덱스마다
    3×224×224 텐서를 지연 생성하는 모듈 수준 클래스(SmokeImageDataset, Windows spawn 피클링용)다. 192×8=1536개라
    두 로더 모두 정확히 8배치다. ⑤ ResNet50Encoder(pretrained=False)는 cuda 전용, ⑥ 합성 2런은 그 사본에서
    --force로 학습하고 기준본(컨테이너) 또는 원본 트리(호스트)의 metrics.json과 비교한다(±0.01/±0.02, 넓히지 않음).
 8. fetch는 --from을 읽기 전용으로 다룬다(복사하면서 sha256을 재고, 일치할 때만 이름을 바꾼다. 원본에서 풀지 않는다).
    내려받기도 <파일>.part에 받은 뒤 sha256이 맞을 때만 이름을 바꾼다. tar는 extractall(filter='data').
    데이터 무결성은 torchvision 자체의 md5 표(CIFAR*.train_list·test_list·meta, SVHN.split_list['test'])와
    download=False 생성자로 자식 프로세스에서 확인한다(네트워크 없음). --from에 압축본이 없고 추출본만 있으면
    추출본 폴더를 복사하고 같은 md5 표로 검증한다(복사 자체는 REPORT, 판정은 md5 검사). 자식 결과는 종합 검사
    'torchvision 무결성'으로도 판정한다: 자식 종료코드 ≠ 0, ok ≠ True, error, md5 11건·생성자 3종 미달이면 FAIL
    (torchvision import가 깨져 항목 검사가 0건인 경우의 공허한 PASS 방지). 자식 실행은 보고서 step으로도 남긴다.
    추출본을 복사했는데 --skip-integrity(시험용 숨은 옵션)이면 검증된 것이 없으므로 FAIL이다.
 9. train·reproduce는 nmfc_pipeline.run_pipeline을 부르고, runs/pipeline_status.json의 항목을 보고서로 옮긴다
    (실행한 항목 = step, 건너뛴 항목 = N/A 검사). train에 --stage·--exp가 모두 없으면 학습 T1~T8 전체다.
    train·reproduce 둘 다 --force(캐시 무시)와 --dry-run(계획만 출력, 아무것도 쓰지 않음)이 있다.
10. compare의 종료코드는 retrain_compare()['exit_code'] 그대로다(전제조건 불충족·미평가 = 2).
    run_pipeline은 호출마다 runs/pipeline_status.json을 새로 쓴다. 그래서 `train --stage T1 --force` 뒤에
    `train --exp c100v_shrb …`를 하면 마지막 상태만 남아 합성 9런 게이트가 빠진다. 이를 막으려고 train·reproduce는
    실행 전에 기존 상태를 runs/pipeline_history/pipeline_status.<started_ns>.json으로 보관하고, compare는 보관본과
    현재 상태를 시작 시각 순으로 합친다(items·trained 합집합, 최상위 시작 = 가장 이른 시작). 신선도는 여전히 항목별
    시작 시각(started_ns)으로 판정된다. --latest-only면 현재 상태만 쓴다. 볼륨(runs)을 지우면 이력도 사라진다.
    합친 상태의 'runs'(실행별 원래 상태)로 retrain_compare가 실행 성공을 판정한다: 마지막 실행이 실패·중단이면
    이력이 있어도 PASS가 아니라 exit 2이고, 이전 실행의 실패는 뒤의 성공 실행이 그 런·단계를 다시 해냈을 때만
    해소된다. 보고서에는 실행마다 '실행 i/n' 검사(pass·exit_code·학습/기대 런·실패 항목·출처 파일)를 넣는다.
11. 보고서의 pass는 종료코드와 어긋나지 않게 한다. 종료코드가 0이 아닌데 FAIL 검사가 없으면
    '종료코드' FAIL 검사를 하나 더한다(예: exit 2인데 pass true인 보고서를 막는다).
12. 판정 입력 파일 바꾸기(시험·망가뜨린 입력 검사용): verify --expected, regen·compare --tolerances,
    fetch --assets.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True          # 원본 트리에 __pycache__를 만들지 않는다(-B와 같은 효과)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import tarfile  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import urllib.request  # noqa: E402
from pathlib import Path  # noqa: E402

DOCKER_DIR = Path(__file__).resolve().parent
if str(DOCKER_DIR) in sys.path:
    sys.path.remove(str(DOCKER_DIR))
sys.path.insert(0, str(DOCKER_DIR))

import nmfc_common as common  # noqa: E402

CLI_PATH = Path(__file__).resolve()
REPO_ROOT: Path = common.REPO_ROOT
EXPECTED_YAML = DOCKER_DIR / "expected.yaml"
TOLERANCES_YAML = DOCKER_DIR / "tolerances.yaml"
ASSETS_JSON = DOCKER_DIR / "assets.json"
STATUS_REL = "runs/pipeline_status.json"
HISTORY_REL = "runs/pipeline_history"
HOST_TRAIN_BASELINE_REL = "runs/nmfc_baseline"
JSON_MARK = "NMFC_JSON="

STEP_TIMEOUT_S = 3600.0          # 단계 하나(검증·집계)의 상한. 걸린 단계를 영원히 기다리지 않는다
SMOKE_TIMEOUT_S = 900.0
TRAIN_SMOKE_TIMEOUT_S = 1800.0
SHM_MIN_BYTES = 2 * 1024 ** 3     # smoke ③
SMOKE_BATCH, SMOKE_BATCHES, SMOKE_WORKERS, SMOKE_IMG = 192, 8, 4, 224
SMOKE_SYN_EXPS = ("syn_moons_softmax", "syn_moons_nmfc")
SMOKE_SYN_TOL = {"test_top1": 0.01, "silhouette": 0.02}

# regen 실행 순서(nmfc_pipeline.STEP_BY_ID의 id). ckpt 단계는 사전조건이 안 맞으면 N/A.
REGEN_ORDER = ("S3.retrieval", "S3.fewshot_verify", "S3.stats_ci", "S3.near_ood", "S3.seed_robustness",
               "S1.bank_softmax", "S1.bank_shrb", "S3.calibration", "S3.openset_detection", "S3.noise_grid")
# assets.json data 항목 → torchvision 추출 폴더(추출본만 있는 --from 처리용, torchvision base_folder와 같다)
EXTRACTED_DIR = {"cifar-100-python.tar.gz": "cifar-100-python", "cifar-10-python.tar.gz": "cifar-10-batches-py"}


class Precondition(Exception):
    """전제조건 불충족(exit 2)."""


# ============================================================================ 공통 도우미

def _say(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(msg.encode(enc, "replace").decode(enc, "replace"), flush=True)


def _compare_mod():
    import nmfc_compare  # noqa: PLC0415 (numpy·yaml을 끌어오므로 지연 import)
    return nmfc_compare


def _pipeline_mod():
    import nmfc_pipeline  # noqa: PLC0415
    return nmfc_pipeline


def _manifest_mod():
    import nmfc_manifest  # noqa: PLC0415
    return nmfc_manifest


def _tables_mod():
    import nmfc_tables  # noqa: PLC0415 (표준 라이브러리만 쓰지만 다른 nmfc_ 모듈과 같은 지연 import)
    return nmfc_tables


def _load_yaml(path: Path) -> dict:
    return _compare_mod().load_yaml(Path(path))


def _same_path(a: Path, b: Path) -> bool:
    try:
        return os.path.normcase(str(Path(a).resolve())) == os.path.normcase(str(Path(b).resolve()))
    except OSError:
        return False


def _mtime_ns(p: Path) -> int | None:
    try:
        return Path(p).stat().st_mtime_ns
    except OSError:
        return None


def _out_dir(a) -> Path:
    return Path(a.out).resolve() if getattr(a, "out", None) else common.default_out()


def _torch_home() -> Path:
    """torch.hub._get_torch_home과 같은 규칙(torch를 import하지 않으려고 옮겨 적음)."""
    th = os.environ.get("TORCH_HOME")
    if th:
        return Path(os.path.expanduser(th))
    return Path(os.path.expanduser(os.path.join(os.environ.get("XDG_CACHE_HOME", "~/.cache"), "torch")))


def _device_env(device: str | None) -> dict:
    """자식 env 덧씌움. cpu면 CUDA를 숨긴다(smoke·regen)."""
    return {"CUDA_VISIBLE_DEVICES": "-1"} if device == "cpu" else {}


def _parse_json_mark(output: str):
    for line in reversed((output or "").splitlines()):
        if line.startswith(JSON_MARK):
            try:
                return json.loads(line[len(JSON_MARK):])
            except ValueError:
                return None
    return None


def _child(args: list[str], *, env: dict | None = None, cwd: Path | None = None, timeout: float = 600.0,
           name: str = "child", echo: bool = False) -> tuple[common.StepResult, object]:
    """이 CLI의 숨은 하위 명령을 자식으로 실행하고 (StepResult, JSON 결과)를 돌려준다."""
    r = common.run_step(name, [str(CLI_PATH), *args], Path(cwd or tempfile.gettempdir()), env=env,
                        timeout=timeout, echo=echo)
    return r, _parse_json_mark(r.output)


def _child_device_info(extra_env: dict | None = None) -> dict:
    """자식 프로세스가 실제로 보는 장치 정보(device_info + python·torchvision). 실패하면 {'error': ...}."""
    r, data = _child(["_device-info"], env=extra_env, timeout=300.0, name="device_info")
    if isinstance(data, dict):
        return data
    return {"device": None, "error": f"_device-info 실패(exit {r.exit_code}): " + " | ".join(r.output_tail[-5:])}


def _hash_tree(root: Path) -> dict[str, str]:
    """{저장소 상대 POSIX 경로: LF 정규화 sha256} — list_files(SKIP_DIRS 제외) 전체."""
    root = Path(root)
    return {rel: common.norm_sha256(root / rel) for rel in common.list_files(root)}


def unexpected_changes(pre: dict[str, str], post: dict[str, str], allowed) -> list[dict]:
    """실행 전후 전체 해시 비교. allowed(writes 경로)의 changed·extra만 허용하고 나머지는 불일치로 낸다."""
    allowed = set(allowed or ())
    out: list[dict] = []
    for rel, sha in pre.items():
        if rel not in post:
            out.append({"path": rel, "kind": "missing", "expected": sha, "actual": None})
        elif post[rel] != sha and rel not in allowed:
            out.append({"path": rel, "kind": "changed", "expected": sha, "actual": post[rel]})
    for rel, sha in post.items():
        if rel not in pre and rel not in allowed:
            out.append({"path": rel, "kind": "extra", "expected": None, "actual": sha})
    out.sort(key=lambda f: (f["path"], f["kind"]))
    return out


def filter_lock_findings(findings: list[dict], allowed) -> list[dict]:
    """verify_lock 결과에서 writes 경로의 changed·extra를 뺀다(missing은 남긴다)."""
    allowed = set(allowed or ())
    return [f for f in findings if not (f["kind"] in ("changed", "extra") and f["path"] in allowed)]


def font_assert(fa: dict) -> tuple[str, str, dict]:
    """한글 폰트 단언. families를 차례로 findfont(fallback_to_default=False)로 찾고,
    처음 찾은(= matplotlib가 실제로 쓰는) 폰트 파일에 char 글리프가 있어야 한다. (status, detail, data)."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    from matplotlib import font_manager as fm  # noqa: PLC0415
    from matplotlib.ft2font import FT2Font  # noqa: PLC0415

    families = list(fa.get("families") or [])
    ch = str(fa.get("char") or "한")
    found = []
    for fam in families:
        try:
            path = fm.findfont(fm.FontProperties(family=fam), fallback_to_default=False)
        except ValueError:
            continue
        try:
            idx = int(FT2Font(path).get_char_index(ord(ch)))
        except Exception as e:  # 폰트 파일을 열 수 없음
            idx = -1
            found.append({"family": fam, "path": path, "glyph_index": idx, "error": repr(e)})
            continue
        found.append({"family": fam, "path": path, "glyph_index": idx})
    data = {"families": families, "char": ch, "found": found}
    if not found:
        return "FAIL", f"{families} 중 찾은 폰트가 없다(두부 글자가 생긴다)", data
    first = found[0]
    if first["glyph_index"] <= 0:
        return "FAIL", f"1순위 폰트 {first['family']}({first['path']})에 {ch!r} 글리프가 없다", data
    return "PASS", f"{first['family']} ({first['path']}) — {ch!r} 글리프 index {first['glyph_index']}", data


def _write_json(path: Path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(common._jsonable(obj), fh, ensure_ascii=False, indent=1, default=common._json_default)
        fh.write("\n")
    os.replace(tmp, path)
    return path


def _write_tables(rep: common.Report, a, command: str, build) -> None:
    """결과표를 <out>/<command>/에 쓴다. build(dest) → nmfc_tables.Table 목록.

    표마다 건수 줄을 출력하고 위치를 rep.env['result_tables']에 남긴다(검사는 더하지 않는다). 결과표는 판정이
    아니므로 실패해도 종료코드를 바꾸지 않는다: 출력하고 WARN 검사로만 남긴다."""
    dest = _out_dir(a) / command
    try:
        tables = list(build(dest) or [])
    except Exception as e:
        msg = f"결과표 쓰기 실패: {type(e).__name__}: {e}"
        _say(f"[nmfc] {msg}")
        rep.add_check("결과표", "WARN", msg, {"dir": str(dest), "traceback": traceback.format_exc()[-3000:]})
        return
    for t in tables:
        _say(f"[nmfc] 결과표 {t.name}: {t.count_line} → {t.csv}")
    rep.env["result_tables"] = {"dir": str(dest), "tables": [{"name": t.name, "csv": str(t.csv), "md": str(t.md),
                                                              "counts": t.count_line} for t in tables]}


def _base_env(a, extra: dict | None = None) -> dict:
    env = {"python": sys.executable, "python_version": platform.python_version(), "platform": sys.platform,
           "in_container": common.in_container(), "repo_root": str(REPO_ROOT), "out": str(_out_dir(a)),
           "argv": list(getattr(a, "_argv", []) or [])}
    if extra:
        env.update(extra)
    return env


def finish(rep: common.Report, code: int, *, reason: str = "") -> int:
    """보고서를 쓰고 종료코드를 돌려준다. pass가 종료코드와 어긋나지 않게 맞춘다(세부 11)."""
    if code == 0 and not rep.passed:
        code = 1
    if code != 0 and rep.passed:
        rep.add_check("종료코드", "FAIL", reason or f"exit {code}")
    if rep.label is None:
        rep.set_label({0: "PASS", 1: "FAIL", 2: "INCONCLUSIVE", 130: "INTERRUPTED"}.get(code, "FAIL"))
    try:
        jp, mp = rep.write()
        _say(f"[nmfc] 보고서: {jp}")
        _say(f"[nmfc]         {mp}")
    except OSError as e:
        _say(f"[nmfc] 보고서 쓰기 실패: {e!r}")
        if code == 0:
            code = 1
    _say(f"[nmfc] {rep.command}: {'PASS' if code == 0 else 'FAIL' if code == 1 else f'exit {code}'}"
         f" (라벨 {rep.label})")
    return code


class Workspace:
    """작업 트리·기준본·임시 폴더의 생성과 정리(세부 3·4)."""

    def __init__(self, a, *, writes_tree: bool):
        self.a = a
        self.writes_tree = writes_tree
        self.keep = bool(getattr(a, "keep", False))
        self.workdir: Path | None = None
        self.created = False
        self.tmpdirs: list[Path] = []

    # ---- 작업 트리
    def open_workdir(self) -> Path:
        a = self.a
        if getattr(a, "workdir", None):
            wd = Path(a.workdir).resolve()
            if not wd.is_dir():
                raise Precondition(f"--workdir가 없다: {wd}")
            if self.writes_tree and not common.in_container() and _same_path(wd, REPO_ROOT):
                raise Precondition(f"--workdir가 원본 저장소다({wd}). 호스트에서는 원본에 쓰지 않는다 — "
                                   "--workdir를 빼서 일회용 사본을 쓰거나 사본 경로를 주라")
            self.workdir = wd
        elif common.in_container():
            self.workdir = common.make_workdir(REPO_ROOT, keep=True)      # 컨테이너: 그대로 반환
        else:
            t0 = time.monotonic()
            _say(f"[nmfc] 일회용 사본을 만든다: {REPO_ROOT} → {tempfile.gettempdir()}")
            self.workdir = common.make_workdir(REPO_ROOT, keep=self.keep)
            self.created = True
            _say(f"[nmfc] 사본 {self.workdir} ({time.monotonic() - t0:.1f}s)")
        return self.workdir

    # ---- 임시 폴더
    def tempdir(self, prefix: str) -> Path:
        d = Path(tempfile.mkdtemp(prefix=f"{prefix}{os.getpid()}-"))
        if not str(d).isascii():
            _say(f"[nmfc] 경고: 임시 폴더 경로에 비ASCII 문자가 있다: {d} (TEMP/TMPDIR를 ASCII 경로로)")
        self.tmpdirs.append(d)
        return d

    # ---- 기준본
    def run_baseline(self, wd: Path) -> Path:
        """verify·regen의 기준본: --baseline → 컨테이너 $NMFC_BASELINE → 호스트는 지금 wd의 스냅샷."""
        if getattr(self.a, "baseline", None):
            p = Path(self.a.baseline).resolve()
            if not p.is_dir():
                raise Precondition(f"--baseline이 없다: {p}")
            return p
        if common.in_container():
            p = common.default_baseline()
            if p is None or not p.is_dir():
                raise Precondition(f"컨테이너 기준본이 없다: {p} ($NMFC_BASELINE)")
            return p
        d = self.tempdir("nmfc-baseline-")
        n = _manifest_mod().make_baseline(wd, d)
        if n == 0:
            raise Precondition(f"기준본 대상 파일이 0개다: {wd}")
        _say(f"[nmfc] 기준본 스냅샷 {n}개 파일 → {d}")
        return d

    def compare_baseline(self, wd: Path) -> Path:
        """compare의 기준본: --baseline → 컨테이너 $NMFC_BASELINE → <wd>/runs/nmfc_baseline → 원본 저장소 스냅샷."""
        if getattr(self.a, "baseline", None) or common.in_container():
            return self.run_baseline(wd)
        own = wd / HOST_TRAIN_BASELINE_REL
        if own.is_dir() and any(own.iterdir()):
            return own
        d = self.tempdir("nmfc-baseline-")
        n = _manifest_mod().make_baseline(REPO_ROOT, d)
        if n == 0:
            raise Precondition(f"기준본 대상 파일이 0개다: {REPO_ROOT}")
        _say(f"[nmfc] 기준본: {own}이 없어 원본 저장소 스냅샷({n}개 파일)을 쓴다 → {d}")
        return d

    def close(self) -> None:
        if self.workdir is not None and self.created:
            if self.keep:
                _say(f"[nmfc] --keep: 작업 사본을 남긴다 → {self.workdir}")
            common.cleanup_workdir(self.workdir, self.keep)
        for d in self.tmpdirs:
            if self.keep:
                _say(f"[nmfc] --keep: 임시 폴더를 남긴다 → {d}")
            else:
                shutil.rmtree(d, ignore_errors=True)


# ============================================================================ env

def _writable(d: Path) -> tuple[bool, str]:
    d = Path(d)
    if not d.is_dir():
        return False, "디렉터리 없음"
    try:
        fd, p = tempfile.mkstemp(prefix=".nmfc-write-test.", dir=str(d))
        os.close(fd)
        os.unlink(p)
        return True, "쓰기 가능"
    except OSError as e:
        return False, f"쓰기 불가: {e}"


def _shm_bytes() -> int | None:
    if not sys.platform.startswith("linux") or not os.path.isdir("/dev/shm"):
        return None
    st = os.statvfs("/dev/shm")
    return int(st.f_frsize * st.f_blocks)


def env_checks(rep: common.Report, a, *, require_gpu: bool, device_env: dict | None = None) -> dict:
    """env의 단언과 출력. smoke ①에서도 쓴다. 장치 정보 dict를 돌려준다."""
    info = _child_device_info(device_env)
    rep.add_check("env: python", "REPORT", f"{platform.python_version()} ({sys.executable})",
                  {"version": platform.python_version()})
    if info.get("torch"):
        rep.add_check("env: torch", "PASS",
                      f"torch {info.get('torch')} / torchvision {info.get('torchvision')} / CUDA {info.get('cuda')}"
                      f" / cuDNN {info.get('cudnn')}", info)
    else:
        rep.add_check("env: torch", "FAIL", f"torch를 쓸 수 없다: {info.get('error')}", info)
    if info.get("device") == "cuda":
        mem = info.get("gpu_mem")
        rep.add_check("env: GPU", "PASS" if require_gpu else "REPORT",
                      f"{info.get('gpu_name')} cc {info.get('capability')} "
                      f"{(mem or 0) / 1024 ** 3:.1f} GiB", info)
    else:
        rep.add_check("env: GPU", "FAIL" if require_gpu else "REPORT",
                      "CUDA 장치 없음" + (" — GPU가 필요하다" if require_gpu else ""), info)
    shm = _shm_bytes()
    rep.add_check("env: /dev/shm", "REPORT" if shm is not None else "N/A",
                  f"{shm / 1024 ** 3:.2f} GiB" if shm is not None else "Linux 전용 항목", {"bytes": shm})
    try:
        fa = _load_yaml(Path(getattr(a, "expected", None) or EXPECTED_YAML))["verify"]["font_assert"]
        st, detail, data = font_assert(fa)
    except Exception as e:
        st, detail, data = "FAIL", f"한글 폰트 단언을 실행하지 못했다: {e!r}", None
    rep.add_check("env: 한글 폰트", st, detail, data)
    if hasattr(os, "getuid"):
        rep.add_check("env: uid", "REPORT", f"uid {os.getuid()} gid {os.getgid()}")
    if common.in_container():
        dirs = [REPO_ROOT / "results", Path("/cache/torch"), _out_dir(a)]
    else:      # 호스트: 원본 저장소에는 시험 파일도 만들지 않는다
        dirs = [_out_dir(a), _torch_home(), Path(tempfile.gettempdir())]
    for d in dirs:
        if not common.in_container() and d == _out_dir(a):
            d.mkdir(parents=True, exist_ok=True)
        if not common.in_container() and d == _torch_home() and not d.is_dir():
            rep.add_check(f"env: 쓰기 {d}", "REPORT", "아직 없음(첫 사용 때 만들어진다)")
            continue
        ok, why = _writable(d)
        rep.add_check(f"env: 쓰기 {d}", "PASS" if ok else "FAIL", why)
    labels = Path("/opt/nmfc/env/labels.json")
    if common.in_container():
        if labels.is_file():
            try:
                rep.add_check("env: 이미지 라벨", "REPORT", str(labels), json.loads(labels.read_text("utf-8")))
            except ValueError as e:
                rep.add_check("env: 이미지 라벨", "WARN", f"labels.json 해독 실패: {e}")
        else:
            rep.add_check("env: 이미지 라벨", "WARN", f"{labels} 없음")
    rep.add_check("env: manifest", "REPORT", str(common.manifest_sha256()))
    return info


def cmd_env(a) -> int:
    rep = common.Report("env", _out_dir(a))
    info = env_checks(rep, a, require_gpu=a.require_gpu)
    rep.set_env(_base_env(a, {"device": info, "TORCH_HOME": str(_torch_home())}))
    return finish(rep, 0 if rep.passed else 1)


# ============================================================================ verify

def cmd_verify(a) -> int:
    rep = common.Report("verify", _out_dir(a))
    ws = Workspace(a, writes_tree=True)
    M = _manifest_mod()
    code = 1
    try:
        C = _compare_mod()
        exp_path = Path(a.expected) if a.expected else EXPECTED_YAML
        vcfg = _load_yaml(exp_path)["verify"]
        wd = ws.open_workdir()
        baseline = ws.run_baseline(wd)
        lock = wd / M.LOCK_REL
        use_lock = lock.is_file()
        pre = None if use_lock else _hash_tree(wd)
        rep.set_env(_base_env(a, {"workdir": str(wd), "workdir_created": ws.created, "keep": ws.keep,
                                  "baseline": str(baseline), "expected": str(exp_path),
                                  "unexpected_change_mode": "verify_lock(strict)" if use_lock else
                                  "pre-run full hash snapshot",
                                  "pre_snapshot_files": None if pre is None else len(pre)}))

        st, detail, data = font_assert(vcfg.get("font_assert") or {})
        rep.add_check("한글 폰트 단언", st, detail, data)

        allowed: set[str] = set()
        written: list[dict] = []
        for step in vcfg["steps"]:
            writes = list(step.get("writes") or [])
            rels = [str(w["path"]).replace("\\", "/") for w in writes]
            allowed.update(rels)
            before = M.snapshot(wd, rels)
            r = common.run_step(step["name"], list(step["cmd"]), wd, allowed_exit=step.get("allowed_exit", (0,)),
                                expect=step.get("expect"), timeout=float(step.get("timeout", STEP_TIMEOUT_S)),
                                echo=True)
            rep.add_step(r)
            for f in C.compare_written(baseline, wd, writes, before):
                written.append(f)
                rep.add_check(f"{step['name']}: {f['path']}", f["status"], f["detail"],
                              {"class": f["class"], **(f.get("data") or {})})

        if use_lock:
            raw = M.verify_lock(wd, lock, ignore_extra_under=())
            bad = filter_lock_findings(raw, allowed)
            how = f"context.lock 엄격 대조({len(M.read_lock(lock))}개 파일)"
        else:
            bad = unexpected_changes(pre, _hash_tree(wd), allowed)
            how = f"실행 직전 전체 해시 스냅샷 대조({len(pre)}개 파일)"
        kinds = {k: sum(1 for f in bad if f["kind"] == k) for k in ("changed", "missing", "extra")}
        rep.add_check("예상 밖 변경 0건", "FAIL" if bad else "PASS",
                      f"{how}: 불일치 {len(bad)}건 {kinds}" if bad else f"{how}: writes 밖 변경 0건",
                      {"findings": bad[:200], "n": len(bad), "allowed_writes": sorted(allowed)})
        _write_tables(rep, a, "verify", lambda d: _tables_mod().verify_tables(d, baseline, wd, written))
        code = 0 if rep.passed else 1
    except Precondition as e:
        rep.add_check("전제조건", "FAIL", str(e))
        code = 2
    except KeyboardInterrupt:
        code = 130
    finally:
        ws.close()
    return finish(rep, code)


# ============================================================================ regen

def cmd_regen(a) -> int:
    rep = common.Report("regen", _out_dir(a))
    ws = Workspace(a, writes_tree=True)
    code = 1
    try:
        C, P = _compare_mod(), _pipeline_mod()
        tol_path = Path(a.tolerances) if a.tolerances else TOLERANCES_YAML
        tol = _load_yaml(tol_path)
        reference = tol.get("reference_device") or {}
        dev_env = _device_env(a.device)
        info = _child_device_info(dev_env)
        if info.get("error") and not info.get("torch"):
            raise Precondition(f"장치 정보를 얻지 못했다: {info.get('error')}")
        if a.device == "cuda" and info.get("device") != "cuda":
            raise Precondition("--device cuda인데 CUDA 장치가 없다")
        mode = C.regen_mode(info, reference)
        wd = ws.open_workdir()
        baseline = ws.run_baseline(wd)
        rep.set_env(_base_env(a, {"workdir": str(wd), "workdir_created": ws.created, "keep": ws.keep,
                                  "baseline": str(baseline), "tolerances": str(tol_path), "device_arg": a.device,
                                  "device": info, "mode": mode, "reference_device": reference,
                                  "child_env": dev_env}))
        _say(f"[nmfc] regen: 장치 {info.get('device')} ({info.get('gpu_name')}), 판정 모드 {mode}")

        steps_ok = True
        for sid in REGEN_ORDER:
            step = P.STEP_BY_ID[sid]
            reason = P._check_needs(step, wd)
            if reason:
                rep.add_check(f"{sid}", "N/A", f"측정 불가 — {reason}")
                continue
            posts = list(step.get("post") or [])
            expect = {p["expect"]: int(p.get("min", 1)) for p in posts if "expect" in p}
            fresh = [p["fresh"] for p in posts if "fresh" in p]
            before = {rel: _mtime_ns(wd / rel) for rel in fresh}
            r = common.run_step(sid, list(step["cmd"]), wd, allowed_exit=step["allowed_exit"], expect=expect,
                                env=dev_env, timeout=STEP_TIMEOUT_S, echo=True)
            rep.add_step(r)
            steps_ok = steps_ok and r.ok
            for rel in fresh:
                now = _mtime_ns(wd / rel)
                if now is None or now == before.get(rel):
                    steps_ok = False
                    rep.add_check(f"{sid}: 갱신 {rel}", "FAIL", "파일 없음" if now is None else "다시 쓰이지 않음")

        result = C.regen_compare(baseline, wd, mode, tol)
        label = C.regen_label(result, info, reference) if steps_ok else "FAIL"
        _add_regen_checks(rep, result)
        fp = _write_json(_out_dir(a) / "regen_findings.json",
                         {"mode": mode, "label": label, "device": info, "counts": result.get("counts"),
                          "findings": result.get("findings")})
        rep.add_check("regen 판정", "FAIL" if label == "FAIL" else "PASS",
                      f"{label} (모드 {mode}, 전체 finding: {fp.name})", result.get("counts"))
        _write_tables(rep, a, "regen", lambda d: _regen_tables(d, rep, C, P, baseline, wd, mode, tol, label))
        rep.set_label(label)
        code = 0 if label in ("PASS_EXACT", "PASS_TOLERANCE") and rep.passed else 1
    except Precondition as e:
        rep.add_check("전제조건", "FAIL", str(e))
        code = 2
    except KeyboardInterrupt:
        code = 130
    finally:
        ws.close()
    return finish(rep, code)


def _add_regen_checks(rep: common.Report, result: dict) -> None:
    """FAIL·WARN·N/A·보고 전용(gate False) finding은 하나씩, 허용오차 안 PASS는 파일별 요약으로 넣는다."""
    per_file: dict[str, dict] = {}
    for f in result.get("findings") or []:
        st = f.get("status")
        name = f"{f.get('file')}:{f.get('path') or ''}".rstrip(":")
        if st == "PASS":
            s = per_file.setdefault(f.get("file") or "?", {"n": 0, "max_delta": 0.0})
            s["n"] += 1
            d = f.get("delta")
            if isinstance(d, (int, float)) and not isinstance(d, bool) and math.isfinite(d):
                s["max_delta"] = max(s["max_delta"], abs(float(d)))
            continue
        if st not in ("FAIL", "WARN", "N/A", "REPORT"):
            st = "REPORT"
        rep.add_check(name, st, str(f.get("detail") or ""),
                      {k: f.get(k) for k in ("rule", "check", "ref", "new", "delta", "tol", "gate") if k in f})
    for file, s in sorted(per_file.items()):
        rep.add_check(f"{file}: 허용오차 안 비동일", "PASS", f"{s['n']}개 리프, max|Δ|={s['max_delta']:.3g}", s)


def _regen_tables(dest: Path, rep: common.Report, C, P, baseline: Path, wd: Path, mode: str, tol: dict,
                  label: str) -> list:
    """regen 결과표: 같은 값까지 담은 행(regen_compare include_identical)에 단계 상태를 붙여 nmfc_tables에 넘긴다.
    건너뛴 단계(이름이 단계 id인 N/A 검사)와 다시 쓰이지 않은 파일('<단계 id>: 갱신 <경로>' FAIL 검사)을 cmd_regen이
    남긴 검사에서 읽는다. 판정(findings·라벨)은 바꾸지 않는다."""
    skipped: dict = {}
    stale: dict = {}
    for c in rep.checks:
        name, st = str(c.get("name") or ""), c.get("status")
        if st == "N/A" and name in REGEN_ORDER:
            posts = P.STEP_BY_ID[name].get("post") or []
            skipped[name] = {"reason": c.get("detail") or "", "outputs": [p["fresh"] for p in posts if "fresh" in p]}
        elif st == "FAIL" and ": 갱신 " in name:
            sid, rel = name.split(": 갱신 ", 1)
            stale[rel] = f"{sid}: {c.get('detail') or ''}"
    rows = C.regen_compare(baseline, wd, mode, tol, include_identical=True).get("rows") or []
    return _tables_mod().regen_tables(dest, rows, skipped=skipped, stale=stale, label=label, mode=mode)


# ============================================================================ fetch

def _copy_hash(src: Path, dst: Path) -> tuple[Path, int, str]:
    """src를 dst.part로 복사하면서 sha256을 잰다. (part 경로, 크기, sha256)."""
    part = dst.with_name(dst.name + ".part")
    h, n = hashlib.sha256(), 0
    with open(src, "rb") as fi, open(part, "wb") as fo:
        for chunk in iter(lambda: fi.read(1 << 20), b""):
            fo.write(chunk)
            h.update(chunk)
            n += len(chunk)
    return part, n, h.hexdigest()


def _download(url: str, dst: Path, size_hint: int | None, timeout: float = 60.0) -> tuple[Path, int, str]:
    part = dst.with_name(dst.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "nmfc-fetch/1 (+https://github.com/)"})
    h, n, last = hashlib.sha256(), 0, 0
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(part, "wb") as fo:
        total = int(resp.headers.get("Content-Length") or 0) or size_hint or 0
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fo.write(chunk)
            h.update(chunk)
            n += len(chunk)
            if n - last >= 32 << 20:
                last = n
                pct = f" {100 * n / total:.0f}%" if total else ""
                _say(f"[fetch]   {dst.name}: {n / 1e6:.0f} MB{pct}")
    return part, n, h.hexdigest()


def _file_ok(p: Path, it: dict) -> tuple[bool, int | None, str | None]:
    if not p.is_file():
        return False, None, None
    sha = common.sha256_file(p)
    size = p.stat().st_size
    return (sha == it["sha256"] and (it.get("size") is None or size == int(it["size"]))), size, sha


def _accept(part: Path, target: Path, n: int, sha: str, it: dict) -> tuple[bool, str]:
    want_size = it.get("size")
    if sha != it["sha256"] or (want_size is not None and n != int(want_size)):
        try:
            part.unlink()
        except OSError:
            pass
        return False, f"sha256/크기 불일치(받음 {sha[:12]}… {n}B, 기대 {it['sha256'][:12]}… {want_size}B)"
    os.replace(part, target)
    return True, f"sha256 일치({sha[:12]}…, {n}B)"


def extract_tar(archive: Path, dest: Path) -> list[str]:
    """tar.gz를 dest에 푼다(filter='data': 절대경로·상위 경로·링크 탈출·장치 파일 거부). 최상위 이름 목록."""
    with tarfile.open(archive, "r:gz") as tf:
        tops = sorted({m.name.split("/", 1)[0] for m in tf.getmembers() if m.name})
        tf.extractall(dest, filter="data")
    return tops


def _resolve_asset_dest(it: dict, what: str, dest_override: str | None) -> Path:
    if dest_override:
        return Path(dest_override).resolve() / it["name"]
    d = str(it.get("dest") or "")
    if d.startswith("$TORCH_HOME/"):
        return _torch_home() / d[len("$TORCH_HOME/"):]
    if what == "data":
        return (REPO_ROOT / d) if d else (REPO_ROOT / "data" / "raw" / it["name"])
    return _torch_home() / "hub" / "checkpoints" / it["name"]


# torchvision 0.26 무결성 표: CIFAR-100 train·test·meta 3 + CIFAR-10 data_batch_1~5·test_batch·batches.meta 7
# + SVHN test 1 = md5 11건, download=False 생성자 3종(CIFAR100·CIFAR10 test, SVHN test).
TV_INTEGRITY_FILES = 11
TV_INTEGRITY_DATASETS = 3


def integrity_checks(rep: common.Report, r: common.StepResult, res) -> None:
    """_tv-integrity 자식의 결과를 검사로 옮긴다(fetch 합격 조건: torchvision 무결성 검사를 네트워크 없이 통과).

    항목별 md5·생성자 검사 외에 종합 검사 'torchvision 무결성'을 항상 하나 넣는다. 자식의 실패 신호(종료코드 ≠ 0,
    ok ≠ True, error)나 검사 개수 부족(md5 11건·생성자 3종이 아님) 중 하나라도 있으면 FAIL이다. 그래서 자식이
    torchvision import 단계에서 죽어 항목 검사가 0건이어도 PASS가 되지 않는다.
    """
    if not isinstance(res, dict):
        rep.add_check("torchvision 무결성", "FAIL",
                      f"검사 실행 실패(exit {r.exit_code}): " + " | ".join(r.output_tail[-5:]))
        return
    files = [f for f in res.get("files") or [] if isinstance(f, dict)]
    dsets = [d for d in res.get("datasets") or [] if isinstance(d, dict)]
    for f in files:
        rep.add_check(f"md5 {f.get('path')}", "PASS" if f.get("ok") else "FAIL",
                      f"torchvision md5 {f.get('md5')}" + ("" if f.get("ok") else " 불일치 또는 없음"))
    for d in dsets:
        rep.add_check(f"torchvision {d.get('name')}", "PASS" if d.get("ok") else "FAIL",
                      f"download=False 생성 성공, len={d.get('len')}" if d.get("ok") else f"실패: {d.get('error')}")
    problems = []
    if res.get("error"):
        problems.append(f"검사 중 예외: {res['error']}")
    if r.exit_code != 0:
        problems.append(f"자식 종료코드 {r.exit_code}")
    if res.get("ok") is not True:
        problems.append(f"자식 판정 ok={res.get('ok')!r}")
    if len(files) != TV_INTEGRITY_FILES:
        problems.append(f"md5 검사 {len(files)}건(기대 {TV_INTEGRITY_FILES}건)")
    if len(dsets) != TV_INTEGRITY_DATASETS:
        problems.append(f"생성자 검사 {len(dsets)}종(기대 {TV_INTEGRITY_DATASETS}종)")
    n_bad = sum(1 for f in files if not f.get("ok")) + sum(1 for d in dsets if not d.get("ok"))
    if n_bad:
        problems.append(f"개별 검사 실패 {n_bad}건")
    rep.add_check("torchvision 무결성", "FAIL" if problems else "PASS",
                  "; ".join(problems) if problems
                  else f"md5 {len(files)}건·생성자 {len(dsets)}종 모두 통과(자식 exit 0)",
                  {"exit_code": r.exit_code, "ok": res.get("ok"), "error": res.get("error"),
                   "files": len(files), "datasets": len(dsets)})


def cmd_fetch(a) -> int:
    rep = common.Report(f"fetch {a.what}", _out_dir(a))
    code = 1
    try:
        assets_path = Path(a.assets) if a.assets else ASSETS_JSON
        assets = json.loads(Path(assets_path).read_text(encoding="utf-8"))
        items = list(assets.get(a.what) or [])
        if not items:
            raise Precondition(f"{assets_path}에 '{a.what}' 항목이 없다")
        src_dir = Path(a.from_).resolve() if a.from_ else None
        if src_dir is not None and not src_dir.is_dir():
            raise Precondition(f"--from 폴더가 없다: {src_dir}")
        targets = [_resolve_asset_dest(it, a.what, a.dest) for it in items]
        dest_root = targets[0].parent
        rep.set_env(_base_env(a, {"assets": str(assets_path), "from": str(src_dir) if src_dir else None,
                                  "dest": str(dest_root), "TORCH_HOME": str(_torch_home())}))
        extracted_only: list[str] = []
        for it, target in zip(items, targets):
            name = it["name"]
            target.parent.mkdir(parents=True, exist_ok=True)
            ok, how, data = False, "", {"target": str(target)}
            got_ok, size, sha = _file_ok(target, it)
            if got_ok:
                ok, how = True, f"이미 있음 — sha256 일치({sha[:12]}…)"
            elif src_dir is not None:
                src = src_dir / name
                if src.is_file():
                    if _same_path(src, target):
                        ok, how = False, "--from과 대상이 같은 파일인데 sha256이 맞지 않는다"
                    else:
                        _say(f"[fetch] {name}: 복사 {src} → {target}")
                        part, n, sha = _copy_hash(src, target)
                        ok, how = _accept(part, target, n, sha, it)
                        how = f"--from 복사: {how}"
                elif it.get("extract") == "tar.gz" and (src_dir / EXTRACTED_DIR.get(name, "\0")).is_dir():
                    folder = EXTRACTED_DIR[name]
                    _say(f"[fetch] {name}: 압축본 없음 → 추출본 {src_dir / folder} 복사(torchvision md5로 검증)")
                    shutil.copytree(src_dir / folder, target.parent / folder, dirs_exist_ok=True)
                    extracted_only.append(folder)
                    # 복사만으로는 검증된 것이 없다. 판정은 아래 'torchvision 무결성'(md5·생성자)이 한다.
                    rep.add_check(f"{name}", "REPORT", f"압축본이 없어 추출본 {folder}/를 복사함(판정은 아래 torchvision "
                                                       "무결성 검사)", data)
                    continue
                else:
                    how = f"--from에 {name}도 추출본도 없다"
            else:
                errs = []
                for url in it.get("urls") or []:
                    _say(f"[fetch] {name}: 내려받기 {url}")
                    try:
                        part, n, sha = _download(url, target, it.get("size"))
                    except Exception as e:  # 네트워크·HTTP·TLS 오류 → 다음 URL
                        errs.append(f"{url}: {type(e).__name__}: {e}")
                        try:
                            target.with_name(target.name + ".part").unlink()
                        except OSError:
                            pass
                        continue
                    ok, msg = _accept(part, target, n, sha, it)
                    if ok:
                        how = f"{url}: {msg}"
                        break
                    errs.append(f"{url}: {msg}")
                if not ok:
                    how = "모든 URL 실패: " + " | ".join(errs) if errs else "URL이 없다"
                data["errors"] = errs
            rep.add_check(f"{name}", "PASS" if ok else "FAIL", how, data)
            if ok and it.get("extract") == "tar.gz":
                try:
                    tops = extract_tar(target, target.parent)
                    rep.add_check(f"{name}: 풀기", "PASS", f"filter='data'로 풂 → {', '.join(tops)}",
                                  {"dest": str(target.parent), "top": tops})
                except Exception as e:
                    rep.add_check(f"{name}: 풀기", "FAIL", f"풀기 실패: {type(e).__name__}: {e}")
            elif it.get("extract") not in (None, "", "tar.gz"):
                rep.add_check(f"{name}: 풀기", "FAIL", f"알 수 없는 extract 형식: {it.get('extract')}")

        if a.what == "data":
            if a.skip_integrity:
                # 추출본 복사는 sha256도 md5도 거치지 않는다. 검사를 건너뛰면 검증된 것이 없으므로 합격시키지 않는다.
                rep.add_check("torchvision 무결성", "FAIL" if extracted_only else "N/A",
                              "--skip-integrity" + (f": 추출본 {extracted_only}은 md5 검사로만 검증되는데 건너뛰었다"
                                                    if extracted_only else ""))
            else:
                r, res = _child(["_tv-integrity", "--root", str(dest_root)], timeout=1800.0,
                                name="torchvision 무결성(download=False)", echo=False)
                rep.add_step(r)
                integrity_checks(rep, r, res)
        code = 0 if rep.passed else 1
    except Precondition as e:
        rep.add_check("전제조건", "FAIL", str(e))
        code = 2
    except KeyboardInterrupt:
        code = 130
    return finish(rep, code)


# ============================================================================ smoke

class SmokeImageDataset:
    """smoke ④용 지연 생성 데이터셋: 인덱스 i마다 (3,size,size) float32 텐서와 라벨 i%classes.

    Windows DataLoader는 spawn이라 데이터셋이 피클링된다. 모듈 수준 클래스여야 하고, 텐서를 미리 만들어 두면
    워커마다 ~1GB를 복사하므로 지연 생성한다(워커→본 프로세스 전송은 공유 메모리를 그대로 거친다).
    """

    def __init__(self, n: int, size: int = SMOKE_IMG, classes: int = 100):
        self.n, self.size, self.classes = int(n), int(size), int(classes)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i):
        import torch  # noqa: PLC0415 (워커에서도 이미 import되어 있다)

        if not 0 <= int(i) < self.n:
            raise IndexError(i)
        return torch.full((3, self.size, self.size), float(int(i) % 251) / 251.0), int(i) % self.classes


def _smoke_matmul(torch, dev, res: dict) -> None:
    x = torch.randn(2048, 2048, device=dev)
    y = torch.randn(2048, 2048, device=dev)
    c32 = x @ y
    with torch.autocast("cuda", dtype=torch.bfloat16):
        c16 = x @ y
    torch.cuda.synchronize()
    rel = float(((c16.float() - c32).abs().max() / c32.abs().max()).item())
    res.update(rel_err_bf16_vs_fp32=rel, out_dtype=str(c16.dtype), gpu=torch.cuda.get_device_name(),
               capability=list(torch.cuda.get_device_capability()))
    if c16.dtype != torch.bfloat16:
        raise RuntimeError(f"autocast 결과가 bf16이 아니다: {c16.dtype}")
    if not math.isfinite(rel) or rel > 0.05:
        raise RuntimeError(f"bf16 matmul 오차가 비정상이다: {rel}")


def _smoke_loader(torch, dev, root: Path, res: dict, bs: int, nb: int, workers: int) -> None:
    sys.path.insert(0, str(root))
    from src.datasets import image as image_mod  # noqa: PLC0415 (실제 로더 팩토리)

    ds = SmokeImageDataset(bs * nb)
    res["factories"] = ["src.datasets.image.make_train_loader(mode='shuffle')", "src.datasets.image.make_eval_loader"]
    for name in ("train", "eval"):
        if name == "train":
            dl = image_mod.make_train_loader(ds, "shuffle", batch_size=bs, num_workers=workers, seed=42)
        else:
            dl = image_mod.make_eval_loader(ds, batch_size=bs, num_workers=workers)
        info = {"num_workers": dl.num_workers, "pin_memory": dl.pin_memory,
                "persistent_workers": dl.persistent_workers, "batch_size": bs}
        t0 = time.monotonic()
        n, pinned, shapes = 0, [], set()
        for x, y in dl:
            pinned.append(bool(x.is_pinned()))
            xd = x.to(dev, non_blocking=True)
            yd = y.to(dev, non_blocking=True)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            shapes.add(tuple(xd.shape))
            if yd.shape[0] != xd.shape[0]:
                raise RuntimeError("라벨 배치 크기가 다르다")
            n += 1
        info.update(batches=n, all_pinned=all(pinned) if pinned else None, shapes=sorted(map(list, shapes)),
                    seconds=round(time.monotonic() - t0, 2))
        res[name] = info
        del dl
        if n != nb:
            raise RuntimeError(f"{name} 로더 배치 수 {n} ≠ {nb}")
        if shapes != {(bs, 3, SMOKE_IMG, SMOKE_IMG)}:
            raise RuntimeError(f"{name} 배치 모양이 다르다: {shapes}")
        if dev.type == "cuda" and not all(pinned):
            raise RuntimeError(f"{name} 로더의 pin_memory가 동작하지 않았다(pin 실패)")


def _smoke_resnet(torch, dev, root: Path, res: dict, bs: int, steps: int) -> None:
    sys.path.insert(0, str(root))
    from src.models.encoders import ResNet50Encoder  # noqa: PLC0415

    torch.manual_seed(0)
    model = ResNet50Encoder(embed_dim=128, pretrained=False).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    torch.cuda.reset_peak_memory_stats(dev)
    losses, dtypes = [], set()
    t0 = time.monotonic()
    for _ in range(steps):
        x = torch.randn(bs, 3, SMOKE_IMG, SMOKE_IMG, device=dev).contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model(x)
            loss = z.float().pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        losses.append(float(loss.detach()))
        dtypes.add(str(z.dtype))
    res.update(max_memory_allocated=int(torch.cuda.max_memory_allocated(dev)),
               max_memory_reserved=int(torch.cuda.max_memory_reserved(dev)), losses=losses,
               out_dtypes=sorted(dtypes), batch=bs, steps=steps, seconds=round(time.monotonic() - t0, 2),
               channels_last=True, total_memory=int(torch.cuda.get_device_properties(dev).total_memory))
    if not all(math.isfinite(v) for v in losses):
        raise RuntimeError(f"손실이 유한하지 않다: {losses}")
    if dtypes != {"torch.bfloat16"}:
        raise RuntimeError(f"bf16 autocast 출력이 아니다: {dtypes}")


def _hidden_smoke_child(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="nmfc _smoke-child")
    ap.add_argument("--what", required=True, choices=["matmul", "loader", "resnet"])
    ap.add_argument("--device", required=True, choices=["cuda", "cpu"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--batch-size", type=int, default=SMOKE_BATCH)
    ap.add_argument("--batches", type=int, default=SMOKE_BATCHES)
    ap.add_argument("--workers", type=int, default=SMOKE_WORKERS)
    ap.add_argument("--steps", type=int, default=3)
    a = ap.parse_args(argv)
    res: dict = {"what": a.what, "device": a.device, "ok": False}
    try:
        import torch  # noqa: PLC0415

        res["torch"] = str(torch.__version__)
        if a.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA를 쓸 수 없다")
        dev = torch.device(a.device)
        if a.what == "matmul":
            _smoke_matmul(torch, dev, res)
        elif a.what == "loader":
            _smoke_loader(torch, dev, Path(a.root).resolve(), res, a.batch_size, a.batches, a.workers)
        else:
            _smoke_resnet(torch, dev, Path(a.root).resolve(), res, a.batch_size, a.steps)
        res["ok"] = True
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
        res["traceback"] = traceback.format_exc()[-3000:]
        oom = "out of memory" in str(e).lower()
        res["oom"] = oom
    print(JSON_MARK + json.dumps(res, ensure_ascii=False), flush=True)
    return 0 if res["ok"] else 1


def _ignore_skip_dirs(dirpath, names):
    """copytree ignore(저장소의 하위 폴더 src/·configs/를 복사할 때): 깊이 무관 이름(__pycache__·
    .pytest_cache) **디렉터리**만 뺀다. data/·out/ 같은 최상위 전용 이름은 하위 폴더에서는 빼지 않는다."""
    return {n for n in names if common.is_skipped_dir(n, at_root=False) and os.path.isdir(os.path.join(dirpath, n))}


def _copy_minimal(src_root: Path, dest: Path) -> None:
    for sub in ("src", "configs"):
        if not (src_root / sub).is_dir():
            raise Precondition(f"{src_root / sub}가 없다")
        shutil.copytree(src_root / sub, dest / sub, ignore=_ignore_skip_dirs)


def _smoke_child_check(rep, label: str, what: str, device: str, root: Path, dev_env: dict) -> dict | None:
    r, res = _child(["_smoke-child", "--what", what, "--device", device, "--root", str(root)], env=dev_env,
                    cwd=root, timeout=SMOKE_TIMEOUT_S, name=label, echo=True)
    rep.add_step(r)
    if not isinstance(res, dict):
        tail = " | ".join(r.output_tail[-6:])
        rep.add_check(label, "FAIL", f"자식 프로세스가 결과 JSON 없이 끝남(exit {r.exit_code}): "
                                     f"{tail}")
        return None
    if res.get("ok"):
        return res
    rep.add_check(label, "FAIL", ("OOM: " if res.get("oom") else "") + str(res.get("error")),
                  {k: v for k, v in res.items() if k != "traceback"} | {"traceback": res.get("traceback")})
    return None


def cmd_smoke(a) -> int:
    rep = common.Report("smoke", _out_dir(a))
    ws = Workspace(a, writes_tree=False)
    code = 1
    dev = a.device
    dev_env = _device_env(dev)
    try:
        src_root = Path(a.workdir).resolve() if a.workdir else REPO_ROOT
        if not src_root.is_dir():
            raise Precondition(f"--workdir가 없다: {src_root}")
        if a.baseline:
            ref_root = Path(a.baseline).resolve()
        elif common.in_container() and common.default_baseline() and common.default_baseline().is_dir():
            ref_root = common.default_baseline()
        else:
            ref_root = src_root
        tmp = ws.tempdir("nmfc-smoke-")
        _copy_minimal(src_root, tmp)
        rep.set_env(_base_env(a, {"device_arg": dev, "child_env": dev_env, "smoke_copy": str(tmp),
                                  "source": str(src_root), "reference": str(ref_root), "keep": ws.keep}))

        # ① env 단언
        info = env_checks(rep, a, require_gpu=(dev == "cuda"), device_env=dev_env)
        rep.env["device"] = info
        if dev == "cpu" and info.get("device") not in ("cpu", None):
            rep.add_check("① 장치 강제", "FAIL", f"CUDA_VISIBLE_DEVICES=-1인데 장치가 {info.get('device')}")
        if not rep.passed:
            for k in ("②", "③", "④", "⑤", "⑥"):
                rep.add_check(f"{k} (건너뜀)", "N/A", "① env 단언 실패로 중단")
            raise _Stop()

        # ② matmul·bf16 autocast
        if dev == "cuda":
            res = _smoke_child_check(rep, "② CUDA matmul·bf16 autocast", "matmul", dev, tmp, dev_env)
            if res:
                rep.add_check("② CUDA matmul·bf16 autocast", "PASS",
                              f"bf16 대 fp32 상대오차 {res['rel_err_bf16_vs_fp32']:.2e}, {res.get('gpu')}", res)
        else:
            rep.add_check("② CUDA matmul·bf16 autocast", "N/A", "--device cpu")

        # ③ /dev/shm
        shm = _shm_bytes()
        if shm is None:
            rep.add_check("③ /dev/shm ≥ 2GiB", "N/A", "Linux 전용(Windows·macOS는 해당 없음)")
        else:
            rep.add_check("③ /dev/shm ≥ 2GiB", "PASS" if shm >= SHM_MIN_BYTES else "FAIL",
                          f"{shm / 1024 ** 3:.2f} GiB" + ("" if shm >= SHM_MIN_BYTES else
                                                          " — compose shm_size: 8gb 또는 --shm-size=8g 필요"),
                          {"bytes": shm})

        # ④ 실제 로더 경로
        res = _smoke_child_check(rep, "④ 실제 로더 경로", "loader", dev, tmp, dev_env)
        if res:
            tr, ev = res.get("train") or {}, res.get("eval") or {}
            rep.add_check("④ 실제 로더 경로", "PASS",
                          f"train {tr.get('batches')}배치·eval {ev.get('batches')}배치 × {SMOKE_BATCH}×3×{SMOKE_IMG}²"
                          f" (workers {tr.get('num_workers')}, pin {tr.get('pin_memory')}, pinned {tr.get('all_pinned')},"
                          f" persistent {tr.get('persistent_workers')}) → {dev}", res)

        # ⑤ ResNet50 bf16 fwd/bwd
        if dev == "cuda":
            res = _smoke_child_check(rep, "⑤ ResNet50 bf16 channels_last fwd/bwd", "resnet", dev, tmp, dev_env)
            if res:
                mm = res["max_memory_allocated"]
                rep.add_check("⑤ ResNet50 bf16 channels_last fwd/bwd", "PASS",
                              f"batch {res['batch']} × {res['steps']}스텝, max_memory_allocated "
                              f"{mm / 1024 ** 3:.2f} GiB ({mm} B), 손실 {['%.4g' % v for v in res['losses']]}", res)
                rep.env["max_memory_allocated"] = mm
        else:
            rep.add_check("⑤ ResNet50 bf16 channels_last fwd/bwd", "N/A", "cuda 전용")

        # ⑥ 합성 2런
        _smoke_synthetic(rep, tmp, ref_root, dev_env)
        code = 0 if rep.passed else 1
    except _Stop:
        code = 1
    except Precondition as e:
        rep.add_check("전제조건", "FAIL", str(e))
        code = 2
    except KeyboardInterrupt:
        code = 130
    finally:
        ws.close()
    return finish(rep, code)


class _Stop(Exception):
    pass


def _smoke_synthetic(rep, tmp: Path, ref_root: Path, dev_env: dict) -> None:
    refs = {}
    for e in SMOKE_SYN_EXPS:
        p = ref_root / "results" / e / "metrics.json"
        try:
            refs[e] = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as ex:
            rep.add_check(f"⑥ {e}", "FAIL", f"기준 metrics.json을 읽지 못했다: {p} ({ex})")
    t_start = time.time()
    r = common.run_step("⑥ 합성 2런(--force)", ["-m", "src.train", "--config", "configs/synthetic.yaml", "--only",
                                                 *SMOKE_SYN_EXPS, "--force"], tmp, env=dev_env,
                        timeout=TRAIN_SMOKE_TIMEOUT_S, echo=True)
    rep.add_step(r)
    for e in SMOKE_SYN_EXPS:
        if e not in refs:
            continue
        p = tmp / "results" / e / "metrics.json"
        if not p.is_file() or p.stat().st_mtime < t_start - 1.0:
            rep.add_check(f"⑥ {e}", "FAIL", "이번 실행에서 metrics.json이 쓰이지 않았다")
            continue
        new = json.loads(p.read_text(encoding="utf-8"))
        for m, t in SMOKE_SYN_TOL.items():
            ra, nb = refs[e].get(m), new.get(m)
            if not isinstance(ra, (int, float)) or not isinstance(nb, (int, float)):
                rep.add_check(f"⑥ {e}/{m}", "FAIL", f"값 없음(기준 {ra!r}, 새 {nb!r})")
                continue
            d = abs(float(nb) - float(ra))
            ok = d <= t * (1 + 1e-9)
            rep.add_check(f"⑥ {e}/{m}", "PASS" if ok else "FAIL",
                          f"기준 {ra:.4f} → 새 {nb:.4f}, |Δ|={d:.4f} {'≤' if ok else '>'} {t}",
                          {"ref": ra, "new": nb, "delta": d, "tol": t})


# ============================================================================ train / reproduce

def archive_status(wd: Path) -> Path | None:
    """기존 runs/pipeline_status.json을 runs/pipeline_history/에 보관한다(세부 10). 보관 경로 또는 None."""
    sp = Path(wd) / STATUS_REL
    if not sp.is_file():
        return None
    try:
        st = json.loads(sp.read_text(encoding="utf-8"))
    except ValueError:
        return None
    key = st.get("started_ns") if isinstance(st.get("started_ns"), int) else sp.stat().st_mtime_ns
    dest = Path(wd) / HISTORY_REL / f"pipeline_status.{key}.json"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sp, dest)
    return dest


def merged_status(wd: Path, latest_only: bool = False) -> tuple[dict, list[str]]:
    """현재 상태 + 보관된 이전 상태(시작 시각 순)를 합친다. (상태, 출처 경로 목록).

    합친 상태에는 실행별 원래 상태를 시작 순으로 담은 'runs'(각각 '_source' = 파일 경로)를 붙인다.
    retrain_compare가 이것으로 실행마다 pass·exit_code·실패 항목을 판정한다(마지막 실행 실패는 PASS 불가).
    """
    sp = Path(wd) / STATUS_REL
    cur = json.loads(sp.read_text(encoding="utf-8"))
    if not isinstance(cur, dict):
        raise ValueError(f"{sp}: 상태 JSON이 객체가 아니다")
    cur["_source"] = str(sp)
    if latest_only:
        return cur, [str(sp)]
    hist = []
    hd = Path(wd) / HISTORY_REL
    if hd.is_dir():
        for p in sorted(hd.glob("pipeline_status.*.json")):
            try:
                h = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(h, dict) or (h.get("started_ns") is not None
                                           and h.get("started_ns") == cur.get("started_ns")):
                continue                                    # 현재 상태와 같은 실행
            h["_source"] = str(p)
            hist.append((h.get("started_ns") or 0, p, h))
    if not hist:
        return cur, [str(sp)]
    hist.sort(key=lambda t: t[0])
    items: list = []
    trained: list = []
    for _, _, h in hist:
        items += list(h.get("items") or [])
        trained += list(h.get("trained") or [])
    items += list(cur.get("items") or [])
    trained += list(cur.get("trained") or [])
    merged = dict(cur)
    merged["items"] = items
    merged["trained"] = list(dict.fromkeys(str(t) for t in trained))
    starts = [h["started_ns"] for _, _, h in hist if isinstance(h.get("started_ns"), int)]
    if isinstance(cur.get("started_ns"), int):
        starts.append(cur["started_ns"])
    if starts:
        merged["started_ns"] = min(starts)
    merged["started"] = hist[0][2].get("started") or cur.get("started")
    sources = [str(p) for _, p, _ in hist] + [str(sp)]
    merged["merged_from"] = sources
    merged["runs"] = [h for _, _, h in hist] + [cur]
    return merged, sources


def _flatten_stages(stages) -> list[str] | None:
    if not stages:
        return None
    out: list[str] = []
    for grp in stages:
        for s in (grp if isinstance(grp, (list, tuple)) else [grp]):
            out += [x.strip() for x in str(s).split(",") if x.strip()]
    return out or None


def _status_to_report(rep: common.Report, status: dict) -> None:
    for it in status.get("items") or []:
        name = f"{it.get('id')} {it.get('name') or ''}".strip()
        if it.get("exit") is None:
            rep.add_check(name, "N/A", str(it.get("skipped_reason") or "실행하지 않음"))
            continue
        notes = [f"{'OK' if p.get('ok') else 'FAIL'}: {p.get('cond')}" + (f" ({p['detail']})" if p.get("detail") else "")
                 for p in it.get("post") or []]
        notes += list(it.get("notes") or [])
        if it.get("trained"):
            notes.append(f"이번에 학습: {', '.join(it['trained'])}")
        if it.get("cached"):
            notes.append(f"캐시로 건너뜀: {', '.join(it['cached'])}")
        rep.add_step({"name": name, "cmd": list(it.get("cmd") or []), "exit_code": it.get("exit"),
                      "duration_s": it.get("duration_s") or 0.0, "output_tail": list(it.get("output_tail") or []),
                      "ok": bool(it.get("ok")), "notes": notes})


def _train_table_baseline(a, wd: Path) -> Path | None:
    """train 결과표의 커밋 값 위치: --baseline → 컨테이너 $NMFC_BASELINE → 호스트 <wd>/runs/nmfc_baseline(학습 전 스냅샷)."""
    if getattr(a, "baseline", None):
        return Path(a.baseline).resolve()
    if common.in_container():
        return common.default_baseline()
    return Path(wd) / HOST_TRAIN_BASELINE_REL


def _run_pipeline_cmd(a, command: str) -> int:
    rep = common.Report(command, _out_dir(a))
    ws = Workspace(a, writes_tree=not a.dry_run)
    code = 1
    try:
        P = _pipeline_mod()
        if command == "train":
            stages = _flatten_stages(a.stage)
            exp = a.exp
            if stages is None and exp is None:
                stages = list(P.STAGE_IDS)          # train 인자 없음 = 학습 T1~T8 전체(후속 단계 없음)
            kw = dict(stages=stages, exp=exp, force=a.force, then_aggregate=(a.then == "aggregate"))
        else:
            kw = dict(force=a.force, with_ext=a.with_ext, from_step=a.from_step, to_step=a.to_step)
        if a.dry_run:
            wd = Path(a.workdir).resolve() if a.workdir else REPO_ROOT      # 계획만: 아무것도 쓰지 않는다
            if not wd.is_dir():
                raise Precondition(f"--workdir가 없다: {wd}")
        else:
            wd = ws.open_workdir()
        out = _out_dir(a)
        rep.set_env(_base_env(a, {"workdir": str(wd), "workdir_created": ws.created, "keep": ws.keep,
                                  "pipeline_args": {k: v for k, v in kw.items()}, "dry_run": a.dry_run}))
        if not a.dry_run and not common.in_container():
            if ws.created and not ws.keep:
                _say("[nmfc] 주의: 호스트 일회용 사본에서 실행한다. 끝나면 사본(학습 결과 포함)을 지운다. "
                     "compare로 이어 가려면 --keep 또는 --workdir를 쓰라. 상태·로그는 <out>/reproduce/에 복사된다.")
            bl = wd / HOST_TRAIN_BASELINE_REL
            if not (bl.is_dir() and any(bl.iterdir())):
                n = _manifest_mod().make_baseline(wd, bl)
                _say(f"[nmfc] compare 기준본(학습 전 상태) {n}개 파일 → {bl}")
            rep.env["compare_baseline"] = str(bl)
        if not a.dry_run:
            arch = archive_status(wd)
            if arch:
                _say(f"[nmfc] 이전 상태를 보관했다(compare가 합쳐 판정): {arch}")
                rep.env["archived_previous_status"] = str(arch)
        rc = P.run_pipeline(wd, out=None if a.dry_run else out, dry_run=a.dry_run, **kw)
        if a.dry_run:
            rep.add_check("dry-run", "REPORT" if rc == 0 else "FAIL", f"계획만 출력함(exit {rc})")
            code = rc
        else:
            sp = wd / STATUS_REL
            status = None
            if sp.is_file():
                try:
                    status = json.loads(sp.read_text(encoding="utf-8"))
                except ValueError:
                    status = None
            if status:
                _status_to_report(rep, status)
                rep.env["status"] = {k: status.get(k) for k in ("started", "finished", "exit_code", "pass",
                                                                "queue_parity", "log_dir")}
                trained = list(status.get("trained") or [])
                rep.add_check("이번에 학습한 런", "REPORT", f"{len(trained)}런: {', '.join(trained)}",
                              {"trained": trained})
                if command == "train":
                    _write_tables(rep, a, "train", lambda d: _tables_mod().train_tables(
                        d, _train_table_baseline(a, wd), wd, status))
            else:
                rep.add_check("상태 JSON", "FAIL" if rc != 2 else "N/A", f"{sp} 없음 또는 해독 실패")
            if rc == 0 and not ws.keep and ws.created:
                rep.add_check("작업 사본", "WARN", "일회용 사본은 끝나면 지워진다 — compare에는 --keep/--workdir 필요")
            if not common.in_container() and (ws.keep or not ws.created):
                _say(f"[nmfc] 이어서 비교: python docker/nmfc_cli.py compare --workdir \"{wd}\"")
            code = rc
    except Precondition as e:
        rep.add_check("전제조건", "FAIL", str(e))
        code = 2
    except KeyboardInterrupt:
        code = 130
    finally:
        ws.close()
    return finish(rep, code)


def cmd_train(a) -> int:
    return _run_pipeline_cmd(a, "train")


def cmd_reproduce(a) -> int:
    return _run_pipeline_cmd(a, "reproduce")


# ============================================================================ compare

def _gate_name(g: dict) -> str:
    if g.get("path") and not g.get("metric"):              # file_change 보고
        return f"{g.get('kind', '?')}: {g['path']}"
    who = g.get("exp") or g.get("group") or g.get("family") or ""
    return f"{g.get('kind', '?')}: {who}/{g.get('metric', '')}".rstrip("/")


def cmd_compare(a) -> int:
    rep = common.Report("compare", _out_dir(a))
    ws = Workspace(a, writes_tree=False)
    code = 2
    try:
        C = _compare_mod()
        if a.workdir:
            wd = Path(a.workdir).resolve()
            if not wd.is_dir():
                raise Precondition(f"--workdir가 없다: {wd}")
        else:
            wd = REPO_ROOT          # 컨테이너 /workspace/nmfc(결과 볼륨), 호스트는 원본(읽기만)
        tol_path = Path(a.tolerances) if a.tolerances else TOLERANCES_YAML
        tol = _load_yaml(tol_path)
        sp = wd / STATUS_REL
        if not sp.is_file():
            raise Precondition(f"{sp}가 없다 — 먼저 train/reproduce를 실행하라(호스트: --keep 또는 --workdir)")
        try:
            status, sources = merged_status(wd, latest_only=a.latest_only)
        except (OSError, ValueError) as e:
            raise Precondition(f"{sp} 해독 실패: {e}") from None
        baseline = ws.compare_baseline(wd)
        rep.set_env(_base_env(a, {"workdir": str(wd), "baseline": str(baseline), "tolerances": str(tol_path),
                                  "status": str(sp), "status_sources": sources,
                                  "trained_total": len(status.get("trained") or [])}))
        if len(sources) > 1:
            _say(f"[nmfc] 상태 {len(sources)}개를 합쳐 판정한다(이전 train 실행 포함, --latest-only로 끔)")
        res = C.retrain_compare(baseline, wd, status, tol)
        pre = res.get("precondition") or {}
        runs = res.get("runs") or []
        for i, run in enumerate(runs, 1):
            st = {"ok": "REPORT", "resolved": "WARN"}.get(run.get("state"), "FAIL")
            fails = "; ".join(f"{f['id']}(exit {f.get('exit')}" + (f", 미완료 {f['unfinished']}" if f.get("unfinished")
                                                                    else "") + ")"
                              for f in run.get("failed_items") or [])
            rep.add_check(f"실행 {i}/{len(runs)}" + (" (마지막)" if run.get("latest") else ""), st,
                          f"{run.get('state')}: 시작 {run.get('started')}, pass={run.get('pass')}, "
                          f"exit_code={run.get('exit_code')}, 학습 {len(run.get('trained') or [])}"
                          f"/기대 {len(run.get('expected_to_train') or [])}런"
                          + (f", 실패 {fails}" if fails else "")
                          + (f", 미해소 {run['unresolved']}" if run.get("unresolved") else "")
                          + f" ({run.get('source')})", run)
        rep.add_check("전제조건(실행 성공·신선도·볼륨)", "PASS" if pre.get("ok") else "FAIL",
                      "충족" if pre.get("ok") else "; ".join(pre.get("reasons") or []), pre)
        for g in res.get("gates") or []:
            st = g.get("status") if g.get("status") in ("PASS", "FAIL") else "FAIL"
            rep.add_check(f"gate {_gate_name(g)}", st, str(g.get("detail") or ""),
                          {k: v for k, v in g.items() if k not in ("detail",)})
        for u in res.get("unevaluated") or []:
            rep.add_check(f"미평가 {u.get('exp') or u.get('group') or ''}/{u.get('metric', '')}", "N/A",
                          str(u.get("reason") or ""), u)
        for r_ in res.get("reports") or []:
            st = r_.get("status") if r_.get("status") in ("REPORT", "WARN", "N/A") else "REPORT"
            rep.add_check(f"report {_gate_name(r_)}", st, str(r_.get("detail") or ""),
                          {k: v for k, v in r_.items() if k not in ("detail",)})
        fp = _write_json(_out_dir(a) / "compare_result.json", res)
        _write_tables(rep, a, "compare", lambda d: _tables_mod().compare_tables(d, res))
        code = int(res.get("exit_code", 2))
        summ = res.get("summary") or {}
        rep.add_check("compare 판정", "PASS" if code == 0 else "FAIL",
                      f"exit {code}: {summ.get('reason')} (학습 {summ.get('trained')}런, 게이트 {summ.get('gates')},"
                      f" 미평가 {summ.get('unevaluated')}; 전체: {fp.name})", summ)
        rep.set_label({0: "PASS", 1: "FAIL"}.get(code, "INCONCLUSIVE"))
    except Precondition as e:
        rep.add_check("전제조건", "FAIL", str(e))
        code = 2
    except KeyboardInterrupt:
        code = 130
    finally:
        ws.close()
    return finish(rep, code)


# ============================================================================ 숨은 하위 명령(자식 프로세스용)

def _hidden_device_info(argv: list[str]) -> int:
    info = common.device_info()
    info["python"] = platform.python_version()
    try:
        import torchvision  # noqa: PLC0415

        info["torchvision"] = str(torchvision.__version__)
    except Exception:
        info["torchvision"] = None
    print(JSON_MARK + json.dumps(info, ensure_ascii=False), flush=True)
    return 0


def _hidden_tv_integrity(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="nmfc _tv-integrity")
    ap.add_argument("--root", required=True)
    a = ap.parse_args(argv)
    root = Path(a.root)
    res: dict = {"root": str(root), "files": [], "datasets": [], "ok": False}
    try:
        from torchvision.datasets import CIFAR10, CIFAR100, SVHN  # noqa: PLC0415
        from torchvision.datasets.utils import check_integrity  # noqa: PLC0415

        for cls in (CIFAR100, CIFAR10):
            entries = list(cls.train_list) + list(cls.test_list) + [[cls.meta["filename"], cls.meta["md5"]]]
            for fn, md5 in entries:
                p = root / cls.base_folder / fn
                res["files"].append({"path": f"{cls.base_folder}/{fn}", "md5": md5,
                                     "ok": bool(check_integrity(str(p), md5))})
        _url, fn, md5 = SVHN.split_list["test"]
        res["files"].append({"path": fn, "md5": md5, "ok": bool(check_integrity(str(root / fn), md5))})
        for name, ctor in (("CIFAR100(train=False)", lambda: CIFAR100(str(root), train=False, download=False)),
                           ("CIFAR10(train=False)", lambda: CIFAR10(str(root), train=False, download=False)),
                           ("SVHN(split='test')", lambda: SVHN(str(root), split="test", download=False))):
            try:
                res["datasets"].append({"name": name, "ok": True, "len": len(ctor())})
            except Exception as e:
                res["datasets"].append({"name": name, "ok": False, "error": f"{type(e).__name__}: {e}"})
        res["ok"] = all(f["ok"] for f in res["files"]) and all(d["ok"] for d in res["datasets"])
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
    print(JSON_MARK + json.dumps(res, ensure_ascii=False), flush=True)
    return 0 if res["ok"] else 1


HIDDEN = {"_device-info": _hidden_device_info, "_smoke-child": _hidden_smoke_child,
          "_tv-integrity": _hidden_tv_integrity}


# ============================================================================ argparse

def _common_opts() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    g = p.add_argument_group("공통 옵션")
    g.add_argument("--out", metavar="DIR", help="보고서 위치(기본 $NMFC_OUT, 없으면 <저장소>/out)")
    g.add_argument("--workdir", metavar="DIR",
                   help="이 트리에서 그대로 실행하고 지우지 않는다(기본: 컨테이너 /workspace/nmfc, 호스트 일회용 사본)")
    g.add_argument("--keep", action="store_true", help="호스트 일회용 사본·임시 기준본을 지우지 않고 남긴다")
    g.add_argument("--baseline", metavar="DIR",
                   help="기준본 위치(기본: 컨테이너 $NMFC_BASELINE, 호스트는 실행 직전 스냅샷)")
    return p


def build_parser() -> argparse.ArgumentParser:
    co = _common_opts()
    ap = argparse.ArgumentParser(
        prog="nmfc",
        description="NMFC 논문 실험 재현 도구 — verify·regen(재학습 없음), train/reproduce→compare(재학습). "
                    "종료코드 0=PASS, 1=FAIL, 2=전제조건 불충족. 판정 기준은 docker/expected.yaml(verify)·"
                    "docker/tolerances.yaml(regen·compare). verify·regen·train·compare는 결과표를 <out>/<명령>/에 쓴다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="예: nmfc verify | nmfc regen --device cuda | nmfc fetch data --from /host-data | "
               "nmfc train --stage T1 --force | nmfc compare | "
               "nmfc manifest --verify docker/context.lock --strict\n"
               "호스트(Windows): python docker/nmfc_cli.py <명령> … (일회용 사본 %TEMP%\\nmfc-work-<pid>에서 실행)")
    sub = ap.add_subparsers(dest="cmd", metavar="<명령>")

    p = sub.add_parser("env", parents=[co], help="환경 점검과 단언(Python·torch·CUDA·cuDNN·GPU·shm·폰트·쓰기)")
    p.add_argument("--require-gpu", action="store_true", help="CUDA 장치가 없으면 FAIL")
    p.add_argument("--expected", metavar="YAML", help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_env)

    p = sub.add_parser("verify", parents=[co],
                       help="검증: pytest·audit_theory_claims 실행 → 기대 문자열·다시 쓴 파일 판정·예상 밖 변경 0건")
    p.add_argument("--expected", metavar="YAML", help="expected.yaml 대신 쓸 파일(시험용)")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("regen", parents=[co], help="재생성: 집계 평가기 재실행 → PASS_EXACT|PASS_TOLERANCE|FAIL")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                   help="cpu면 자식에 CUDA_VISIBLE_DEVICES=-1(기본 auto)")
    p.add_argument("--tolerances", metavar="YAML", help="tolerances.yaml 대신 쓸 파일(시험용)")
    p.set_defaults(func=cmd_regen)

    p = sub.add_parser("fetch", parents=[co], help="데이터(data)·ImageNet 가중치(weights) 받기와 sha256 검증")
    p.add_argument("what", choices=["data", "weights"], help="data: CIFAR-100·CIFAR-10·SVHN test, weights: resnet50")
    p.add_argument("--from", dest="from_", metavar="DIR", help="내려받는 대신 DIR의 압축본(또는 추출본)을 복사(읽기만)")
    p.add_argument("--dest", metavar="DIR",
                   help="저장 폴더(기본 data: <저장소>/data/raw, weights: $TORCH_HOME/hub/checkpoints)")
    p.add_argument("--assets", metavar="JSON", help=argparse.SUPPRESS)
    p.add_argument("--skip-integrity", action="store_true", help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("smoke", parents=[co],
                       help="스모크: env·matmul·shm·실제 로더 경로·ResNet50 bf16 fwd/bwd·합성 2런")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="cpu면 CUDA_VISIBLE_DEVICES=-1 강제")
    p.add_argument("--expected", metavar="YAML", help=argparse.SUPPRESS)
    p.set_defaults(func=cmd_smoke)

    p = sub.add_parser("train", parents=[co], help="학습(파이프라인 학습 부분). 인자 없으면 T1~T8 전체")
    p.add_argument("--stage", action="append", nargs="+", metavar="T", help="T1..T8(여럿 가능, 쉼표 구분 가능)")
    p.add_argument("--exp", metavar="NAME", help="큐에 있는 실험 하나만(--only NAME)")
    p.add_argument("--force", action="store_true", help="metrics.json 캐시를 무시하고 다시 학습")
    p.add_argument("--then", choices=["aggregate"],
                   help="학습 뒤 집계: retrieval→fewshot_verify→stats_ci→(ckpt 있을 때 near_ood)→seed_robustness")
    p.add_argument("--dry-run", action="store_true", help="실행할 명령만 출력(아무것도 쓰지 않음)")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("reproduce", parents=[co], help="전체 파이프라인(학습 T1~T8 + 후속 S1~S4)")
    p.add_argument("--with-ext", action="store_true", help="S2 외부 임베딩(ResNet50 동결)도 실행")
    p.add_argument("--from", dest="from_step", metavar="S", help="이 단계부터(T1..T8, S1..S4, 하위 id)")
    p.add_argument("--to", dest="to_step", metavar="S", help="이 단계까지")
    p.add_argument("--force", action="store_true", help="metrics.json 캐시를 무시하고 다시 학습")
    p.add_argument("--dry-run", action="store_true", help="실행할 명령만 출력(아무것도 쓰지 않음)")
    p.set_defaults(func=cmd_reproduce)

    p = sub.add_parser("compare", parents=[co], help="재학습 비교: 신선도·볼륨 가드 → 런별·평균·부호·합성 게이트")
    p.add_argument("--tolerances", metavar="YAML", help="tolerances.yaml 대신 쓸 파일(시험용)")
    p.add_argument("--latest-only", action="store_true",
                   help="마지막 train/reproduce 상태만 판정(기본: runs/pipeline_history의 이전 실행과 합침)")
    p.set_defaults(func=cmd_compare)

    sub.add_parser("manifest", add_help=False,
                   help="context.lock: --write-lock CTX | --verify LOCK [--root R] [--strict] [--json P] "
                        "| --baseline DEST [--root R] (nmfc_manifest로 그대로 넘김)")
    return ap


def main(argv: list[str] | None = None) -> int:
    common.setup_stdio()
    argv = [os.fspath(x) for x in (sys.argv[1:] if argv is None else argv)]
    if argv and argv[0] == "manifest":
        return _manifest_mod().run_manifest_cli(argv)
    if argv and argv[0] in HIDDEN:
        return HIDDEN[argv[0]](argv[1:])
    ap = build_parser()
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 2
    if not getattr(a, "cmd", None):
        ap.print_help()
        return 2
    a._argv = argv
    try:
        return int(a.func(a))
    except KeyboardInterrupt:
        _say("[nmfc] 중단됨")
        return 130


if __name__ == "__main__":
    sys.exit(main())
