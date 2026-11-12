"""NMFC 재현 도구 공통 모듈.

최상위 import는 stdlib만 쓴다. torch는 device_info() 안에서만 지연 import한다.

이 모듈에서 정한 세부
- 텍스트/바이너리 판정: 확장자(소문자)가 BINARY_EXT에 있거나 앞 8000바이트에 NUL이 있으면 바이너리다.
  텍스트만 CRLF→LF로 정규화하고, 단독 CR은 그대로 둔다(git의 text=auto 정규화와 같은 의미).
- 건너뛸 디렉터리(파일 이름은 보지 않는다): SKIP_DIRS 중 __pycache__·.pytest_cache(SKIP_DIRS_ANY_DEPTH)는
  깊이와 상관없이, 나머지(.git·data·checkpoints·out·dist·runs·_planning·.venv·host-data)는 **저장소 최상위**에
  있을 때만 건너뛴다. .venv(scripts/setup_env.ps1이 만드는 호스트 venv)와 host-data(compose fetch 바인드 기본값)는
  .gitignore 대상이라 추적 파일이 없고, 호스트 사본에 수 GB가 복사되는 것을 막으려고 넣었다.
  .dockerignore(`data/` 등은 컨텍스트 최상위만 뺀다)와 같은 의미다. 하위 폴더(results/data/, src/runs/,
  results/<실험>/out/ 등)의 파일은 lock·대조·사본에 그대로 들어가므로 '예상 밖 변경'에서 빠지지 않는다.
  make_workdir 복사와 nmfc_manifest의 lock 작성·검증, verify의 전체 해시가 모두 같은 판정 함수(is_skipped_dir)와
  같은 걷기 함수(list_files)를 쓴다. 저장소의 하위 폴더를 걸을 때는 list_files(..., repo_root=False)로 부른다.
- make_workdir(keep=False)는 프로세스가 끝날 때(atexit) 사본을 지우도록 등록한다. keep=True면 등록하지 않는다.
  cleanup_workdir는 make_workdir가 만든 사본이거나 tempfile.gettempdir() 바로 아래의 `nmfc-work-*`일 때만
  지운다. 그 밖의 경로(원본 저장소, 컨테이너의 /workspace/nmfc, 사용자가 준 --workdir)는 절대 지우지 않는다.
- child_env(extra): os.environ 위에 extra를 덧씌운다(값이 None이면 그 변수를 지운다). 그다음
  PYTHONUTF8=1, PYTHONDONTWRITEBYTECODE=1, PYTHONUNBUFFERED=1, PYTHONIOENCODING=utf-8을 **강제**한다.
  run_step의 env 인자도 같은 의미(덧씌울 변수)다. 완전한 환경으로 바꿔치우지 않는다(Windows에서
  SYSTEMROOT·PATH가 빠지면 파이썬이 깨진다).
- run_step
  - stdout과 stderr를 한 파이프로 합쳐(stderr=STDOUT) UTF-8(errors="replace")로 읽는다. stdin은 DEVNULL이다.
  - expect는 **전체** 출력에서 str.count로 센다. output_tail(마지막 40줄)이 아니다.
  - timeout이 나면 프로세스 트리를 죽인다. Windows는 taskkill /T, POSIX는 새 세션을 만든 뒤 killpg를 쓴다.
    이때 exit_code=-1, ok=False다.
  - 실행 파일을 찾지 못하는 등 시작 실패는 예외를 올리지 않고 exit_code=127, ok=False로 돌려준다.
  - 전체 출력은 데이터클래스 필드가 아닌 속성 `StepResult.output`에 달아 둔다. 그래서 보고서 JSON
    스키마(필드 7개)는 그대로다.
  - echo=True(키워드 전용, 기본 False)면 자식 출력을 실시간으로 sys.stdout에 흘린다. 긴 학습 단계에 쓴다.
- Report
  - status는 {"PASS","FAIL","WARN","REPORT","N/A"} 밖이면 ValueError다.
  - passed는 실패 step 0개이고 FAIL check 0개일 때 True다. 그에 더해 label이 "FAIL"이면 False다.
    보고서가 'label FAIL, pass true'로 모순되지 않게 하려는 것이다.
  - JSON은 UTF-8·LF로 쓴다. Path, set, numpy 스칼라·배열, 데이터클래스는 직렬화 가능한 값으로 바꾼다.
  - .md 첫 줄은 `nmfc.manifest.sha256: <값|none>`이다.
- manifest_sha256(): 빈 문자열 환경변수는 없는 것으로 본다. lock 파일의 sha256은 원시 바이트 기준이다.
  lock은 항상 LF로 쓰므로 sha256sum/Get-FileHash 값과 같다.
- device_info()의 capability는 JSON·YAML과 맞추려고 리스트 [major, minor]로, gpu_mem은 바이트로 준다.
"""
from __future__ import annotations

import atexit
import dataclasses
import hashlib
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent

BINARY_EXT: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".npz", ".npy", ".pt", ".pth", ".pdf", ".pptx", ".zip", ".xlsx", ".mat", ".gz",
})
SKIP_DIRS: frozenset[str] = frozenset({
    ".git", "data", "checkpoints", "out", "dist", "runs", "_planning", "__pycache__", ".pytest_cache",
    ".venv", "host-data",   # scripts/setup_env.ps1의 호스트 venv·compose fetch 바인드 기본값(둘 다 .gitignore, 이미지 밖)
})
SKIP_DIRS_ANY_DEPTH: frozenset[str] = frozenset({"__pycache__", ".pytest_cache"})   # 나머지는 최상위에서만

NUL_SNIFF_BYTES = 8000          # git의 buffer_is_binary와 같은 길이
_CHUNK = 1 << 20                # 해시 스트리밍 단위(1 MiB, NUL_SNIFF_BYTES보다 커야 한다)
TAIL_LINES = 40
VALID_STATUS: tuple[str, ...] = ("PASS", "FAIL", "WARN", "REPORT", "N/A")
CHILD_ENV_FORCED: dict[str, str] = {
    "PYTHONUTF8": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
    "PYTHONIOENCODING": "utf-8",
}


# ---------------------------------------------------------------------------
# 텍스트/바이너리 판정과 LF 정규화 해시
# ---------------------------------------------------------------------------
def _ext(path) -> str:
    if path is None:
        return ""
    return os.path.splitext(os.fspath(path))[1].lower()


def is_binary(path, data: bytes) -> bool:
    """확장자가 BINARY_EXT에 있거나 앞 8000바이트에 NUL이 있으면 바이너리다."""
    return _ext(path) in BINARY_EXT or b"\x00" in data[:NUL_SNIFF_BYTES]


def norm_bytes(path, data: bytes) -> bytes:
    """텍스트면 CRLF→LF(단독 CR은 유지), 바이너리면 그대로 돌려준다."""
    if is_binary(path, data):
        return data
    return data.replace(b"\r\n", b"\n")


def norm_size_sha256(path) -> tuple[int, str]:
    """(정규화한 크기, 정규화한 sha256). 바이너리는 스트리밍으로 해시한다."""
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        head = fh.read(_CHUNK)
        if is_binary(path, head):
            h.update(head)
            size = len(head)
            while True:
                chunk = fh.read(_CHUNK)
                if not chunk:
                    break
                h.update(chunk)
                size += len(chunk)
        else:
            data = (head + fh.read()).replace(b"\r\n", b"\n")
            h.update(data)
            size = len(data)
    return size, h.hexdigest()


def norm_sha256(path: Path) -> str:
    """LF 정규화 sha256(git hash-object와 같은 정규화 의미, 해시는 sha256)."""
    return norm_size_sha256(path)[1]


def sha256_file(path) -> str:
    """원시 바이트 sha256(정규화 없음)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 디렉터리 걷기(make_workdir와 manifest가 공유)
# ---------------------------------------------------------------------------
def is_skipped_dir(name: str, at_root: bool = True) -> bool:
    """건너뛸 디렉터리 이름인가. SKIP_DIRS_ANY_DEPTH는 어디서나, 나머지 SKIP_DIRS는 저장소 최상위(at_root)에서만."""
    if name in SKIP_DIRS_ANY_DEPTH:
        return True
    return at_root and name in SKIP_DIRS


def _raise(err: OSError):
    raise err


def list_files(root: Path, *, repo_root: bool = True) -> list[str]:
    """root 아래 일반 파일의 상대 POSIX 경로를 정렬해 돌려준다(건너뛸 디렉터리 제외, is_skipped_dir 규칙).

    repo_root=True면 root를 저장소 루트로 보고 최상위의 SKIP_DIRS 이름을 뺀다. 저장소의 하위 폴더(results/ 등)를
    걸을 때는 repo_root=False로 불러 그 아래의 data/·out/ 같은 이름을 빼지 않게 한다.
    """
    root = Path(root)
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, onerror=_raise, followlinks=False):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        top = repo_root and rel_dir in ("", ".")
        dirnames[:] = sorted(d for d in dirnames if not is_skipped_dir(d, at_root=top))
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            if not os.path.isfile(full):        # 깨진 링크·특수 파일 제외
                continue
            out.append(fn if rel_dir in ("", ".") else f"{rel_dir}/{fn}")
    out.sort()
    return out


# ---------------------------------------------------------------------------
# 환경
# ---------------------------------------------------------------------------
def _env(name: str) -> str | None:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return None
    return v


def in_container() -> bool:
    return os.environ.get("NMFC_IN_CONTAINER") == "1"


def default_out() -> Path:
    v = _env("NMFC_OUT")
    return Path(v) if v else REPO_ROOT / "out"


def default_baseline() -> Path | None:
    v = _env("NMFC_BASELINE")
    return Path(v) if v else None


def setup_stdio() -> None:
    """stdout/stderr를 UTF-8로 바꾼다(Windows 부모의 기본 인코딩이 cp949라서)."""
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def child_env(extra: dict | None = None) -> dict:
    """os.environ + extra(None 값은 삭제) + UTF-8·바이트코드 금지·무버퍼 강제."""
    env = dict(os.environ)
    for k, v in (extra or {}).items():
        if v is None:
            env.pop(str(k), None)
        else:
            env[str(k)] = str(v)
    env.update(CHILD_ENV_FORCED)
    return env


# ---------------------------------------------------------------------------
# 일회용 작업 사본
# ---------------------------------------------------------------------------
_WORKDIRS: dict[str, bool] = {}     # make_workdir가 만든 사본 → keep 여부
_ATEXIT_REGISTERED = False


def _key(path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def _rmtree(path: Path, ignore_errors: bool = False) -> None:
    path = Path(path)
    if path.is_symlink():
        path.unlink()
        return
    if not path.exists():
        return

    def _onexc(func, p, exc):
        try:
            os.chmod(p, stat.S_IWRITE)      # Windows 읽기 전용 파일
            func(p)
        except FileNotFoundError:
            pass
        except Exception:
            if not ignore_errors:
                raise

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_onexc)
    else:  # pragma: no cover
        shutil.rmtree(path, onerror=lambda f, p, ei: _onexc(f, p, ei[1]))


def _atexit_cleanup() -> None:
    for p, keep in list(_WORKDIRS.items()):
        if not keep:
            try:
                _rmtree(Path(p), ignore_errors=True)
            except Exception:
                pass
            _WORKDIRS.pop(p, None)


def _is_temp_workdir(path: Path) -> bool:
    p = Path(path).resolve()
    return p.name.startswith("nmfc-work-") and _key(p.parent) == _key(tempfile.gettempdir())


def _copy_ignore_for(src_root: Path):
    """copytree ignore: src_root(저장소 루트) 바로 아래면 SKIP_DIRS 전부, 더 깊으면 SKIP_DIRS_ANY_DEPTH만 뺀다."""
    root_key = os.path.normcase(os.path.abspath(os.fspath(src_root)))

    def ignore(dirpath: str, names: list[str]) -> set[str]:
        top = os.path.normcase(os.path.abspath(dirpath)) == root_key
        return {n for n in names if is_skipped_dir(n, at_root=top) and os.path.isdir(os.path.join(dirpath, n))}

    return ignore


def make_workdir(src: Path = REPO_ROOT, keep: bool = False) -> Path:
    """컨테이너면 src를 그대로, 호스트면 tempfile.gettempdir()/nmfc-work-<pid>로 복사해 돌려준다.

    복사에서는 SKIP_DIRS 디렉터리를 뺀다. 같은 경로가 이미 있으면 지우고 새로 만든다.
    keep=False면 프로세스 종료 시 지우도록 등록한다.
    """
    global _ATEXIT_REGISTERED
    src = Path(src)
    if in_container():
        return src
    if not src.is_dir():
        raise FileNotFoundError(f"작업 사본 원본이 없다: {src}")
    dest = Path(tempfile.gettempdir()) / f"nmfc-work-{os.getpid()}"
    src_r, dest_r = src.resolve(), dest.resolve()
    if dest_r == src_r or src_r in dest_r.parents:
        raise ValueError(f"작업 사본 경로가 원본 안에 있다: {dest_r}")
    if not str(dest_r).isascii():
        print(f"[nmfc] 경고: 작업 사본 경로에 비ASCII 문자가 있다: {dest_r} "
              "(TEMP/TMPDIR를 ASCII 경로로 지정하라)", file=sys.stderr)
    if dest.exists() or dest.is_symlink():
        _rmtree(dest)
    shutil.copytree(src, dest, ignore=_copy_ignore_for(src), ignore_dangling_symlinks=True)
    _WORKDIRS[_key(dest)] = bool(keep)
    if not keep and not _ATEXIT_REGISTERED:
        atexit.register(_atexit_cleanup)
        _ATEXIT_REGISTERED = True
    return dest


def cleanup_workdir(path: Path, keep: bool) -> None:
    """keep이면 남긴다. 아니면 make_workdir가 만든 사본(또는 TEMP의 nmfc-work-*)만 지운다."""
    k = _key(path)
    if keep:
        if k in _WORKDIRS:
            _WORKDIRS[k] = True
        return
    if k not in _WORKDIRS and not _is_temp_workdir(Path(path)):
        return      # 우리가 만든 사본이 아니다(원본·컨테이너 작업트리·사용자 지정 경로) → 지우지 않는다
    _rmtree(Path(path))
    _WORKDIRS.pop(k, None)


# ---------------------------------------------------------------------------
# 단계 실행
# ---------------------------------------------------------------------------
@dataclass
class StepResult:
    name: str
    cmd: list[str]
    exit_code: int
    duration_s: float
    output_tail: list[str]       # 출력 마지막 40줄
    ok: bool
    notes: list[str] = field(default_factory=list)

    output = ""                  # 전체 출력(데이터클래스 필드 아님 → 보고서 JSON에 들어가지 않는다)


def _echo(text: str) -> None:
    try:
        sys.stdout.write(text)
        sys.stdout.flush()
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        try:
            sys.stdout.write(text.encode(enc, "replace").decode(enc, "replace"))
            sys.stdout.flush()
        except Exception:
            pass
    except Exception:
        pass


def _pump(stream, sink: list[str], echo: bool) -> None:
    try:
        for line in stream:
            sink.append(line)
            if echo:
                _echo(line)
    except Exception as e:  # 파이프가 닫힌 경우 등
        sink.append(f"\n[nmfc] 출력 읽기 중단: {e!r}\n")
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        else:
            try:
                if os.getpgid(proc.pid) == proc.pid:      # start_new_session으로 만든 그룹일 때만
                    os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
    except Exception:
        pass
    if proc.poll() is None:
        try:
            proc.kill()
        except Exception:
            pass


def _cmd_str(cmd: list[str]) -> str:
    shown = ["python" if i == 0 and c == sys.executable else c for i, c in enumerate(cmd)]
    return " ".join(f'"{c}"' if (" " in c or c == "") else c for c in shown)


def run_step(name: str, args: list[str], cwd: Path, allowed_exit=(0,), expect: dict[str, int] | None = None,
             env: dict | None = None, timeout: float | None = None, python: bool = True, *,
             echo: bool = False) -> StepResult:
    """한 단계를 실행하고 StepResult를 돌려준다.

    python=True면 [sys.executable, "-B", *args], False면 args 그대로 실행한다.
    expect = {부분문자열: 최소 출현 횟수}를 stdout+stderr를 합친 전체 출력에서 센다.
    ok = (exit ∈ allowed_exit) and expect 전부 충족 and 시간 초과 아님.
    """
    cmd = [os.fspath(a) for a in args]
    if python:
        cmd = [sys.executable, "-B", *cmd]
    allowed = tuple(int(x) for x in allowed_exit)
    full_env = child_env(env)
    notes: list[str] = []
    lines: list[str] = []
    popen_kw = dict(cwd=os.fspath(cwd), env=full_env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
    if timeout is not None and os.name == "posix":
        popen_kw["start_new_session"] = True
    if echo:
        _echo(f"[nmfc] >> {name}: {_cmd_str(cmd)}\n")
    t0 = time.monotonic()
    try:
        proc = subprocess.Popen(cmd, **popen_kw)
    except OSError as e:
        notes.append(f"실행 실패: {type(e).__name__}: {e}")
        r = StepResult(name, cmd, 127, round(time.monotonic() - t0, 3), [], False, notes)
        r.output = ""
        if echo:
            _echo(f"[nmfc] << {name}: 실행 실패 ({e})\n")
        return r

    reader = threading.Thread(target=_pump, args=(proc.stdout, lines, echo), daemon=True)
    reader.start()
    timed_out = False
    try:
        try:
            exit_code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_tree(proc)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            exit_code = -1
    except BaseException:           # KeyboardInterrupt 등: 자식을 남기지 않는다
        _kill_tree(proc)
        raise
    reader.join(timeout=10 if timed_out else 60)
    if reader.is_alive():
        notes.append("출력 파이프가 닫히지 않음(자식이 남긴 프로세스가 파이프를 잡고 있다)")
    duration = round(time.monotonic() - t0, 3)
    output = "".join(list(lines))

    ok = True
    if timed_out:
        ok = False
        notes.append(f"시간 초과: {timeout}s 후 프로세스 트리를 강제 종료")
    elif exit_code not in allowed:
        ok = False
        notes.append(f"exit {exit_code} (허용 {list(allowed)})")
    for sub, need in (expect or {}).items():
        need = int(need)
        got = output.count(sub)
        good = got >= need
        ok = ok and good
        notes.append(f"기대 {sub!r} >= {need}회: {got}회" + ("" if good else " (부족)"))

    r = StepResult(name, cmd, int(exit_code), duration, output.splitlines()[-TAIL_LINES:], ok, notes)
    r.output = output
    if echo:
        _echo(f"[nmfc] << {name}: exit {exit_code}, {duration:.1f}s, {'OK' if ok else 'FAIL'}\n")
    return r


# ---------------------------------------------------------------------------
# 보고서
# ---------------------------------------------------------------------------
def _jsonable(o):
    if isinstance(o, dict):
        return {(k if isinstance(k, (str, int, float, bool)) or k is None else str(k)): _jsonable(v)
                for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (set, frozenset)):
        return [_jsonable(v) for v in sorted(o, key=str)]
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return _jsonable(dataclasses.asdict(o))
    return o


def _json_default(o):
    if isinstance(o, Path):
        return o.as_posix()
    if isinstance(o, bytes):
        return o.decode("utf-8", "replace")
    if hasattr(o, "tolist"):          # numpy 배열·스칼라
        return _jsonable(o.tolist())
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return str(o)


def _write_text_atomic(path: Path, text: str) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _md_cell(s, code: bool = False) -> str:
    s = "" if s is None else str(s)
    if not code:                     # 코드 스팬 안에서는 역슬래시 이스케이프가 처리되지 않는다
        s = s.replace("\\", "\\\\")
    return s.replace("|", "\\|").replace("\r", "").replace("\n", "<br>")


def _step_dict(s) -> dict:
    if dataclasses.is_dataclass(s) and not isinstance(s, type):
        return dataclasses.asdict(s)
    return dict(s)


class Report:
    """하위 명령 하나의 보고서. write()가 <out>/<command>_report.{json,md}를 쓴다."""

    def __init__(self, command: str, out_dir: Path):
        self.command = str(command)
        self.out_dir = Path(out_dir)
        self.label: str | None = None
        self.env: dict = {}
        self.steps: list = []
        self.checks: list[dict] = []
        self.started = now_iso()
        self.finished: str | None = None

    def add_step(self, r: StepResult) -> None:
        self.steps.append(r)

    def add_check(self, name: str, status: str, detail: str = "", data=None) -> None:
        if status not in VALID_STATUS:
            raise ValueError(f"check status는 {VALID_STATUS} 중 하나여야 한다: {status!r}")
        self.checks.append({"name": str(name), "status": status, "detail": "" if detail is None else str(detail),
                            "data": data})

    def set_label(self, label: str) -> None:
        self.label = label

    def set_env(self, env: dict) -> None:
        self.env = dict(env or {})

    @property
    def passed(self) -> bool:
        steps_ok = all(bool(_step_dict(s).get("ok")) for s in self.steps)
        checks_ok = not any(c["status"] == "FAIL" for c in self.checks)
        return steps_ok and checks_ok and self.label != "FAIL"

    def to_dict(self) -> dict:
        return {
            "command": self.command,
            "label": self.label,
            "pass": self.passed,
            "manifest_sha256": manifest_sha256(),
            "env": self.env,
            "started": self.started,
            "finished": self.finished,
            "steps": [_step_dict(s) for s in self.steps],
            "checks": list(self.checks),
        }

    def to_markdown(self, d: dict | None = None) -> str:
        d = d if d is not None else self.to_dict()
        L: list[str] = [f"nmfc.manifest.sha256: {d['manifest_sha256'] or 'none'}", ""]
        L.append(f"# nmfc {d['command']} 보고서")
        L.append("")
        L.append(f"- 판정: **{'PASS' if d['pass'] else 'FAIL'}**")
        if d.get("label"):
            L.append(f"- 라벨: **{d['label']}**")
        L.append(f"- 시작: {d['started']} / 종료: {d['finished']}")
        L.append("")
        if d.get("env"):
            L += ["## 환경", "", "| 항목 | 값 |", "|---|---|"]
            for k, v in d["env"].items():
                if not isinstance(v, str):
                    v = json.dumps(_jsonable(v), ensure_ascii=False, default=_json_default)
                L.append(f"| {_md_cell(k)} | {_md_cell(v)} |")
            L.append("")
        L += ["## 단계", ""]
        if d["steps"]:
            L += ["| # | 단계 | 명령 | exit | 소요(s) | 판정 | 메모 |", "|---|---|---|---|---|---|---|"]
            for i, s in enumerate(d["steps"]):
                cmd = _cmd_str([str(c) for c in s.get("cmd", [])])
                L.append(f"| {i} | {_md_cell(s.get('name'))} | `{_md_cell(cmd, code=True)}` | {s.get('exit_code')} | "
                         f"{float(s.get('duration_s') or 0):.1f} | {'OK' if s.get('ok') else 'FAIL'} | "
                         f"{_md_cell('; '.join(s.get('notes') or []))} |")
        else:
            L.append("(없음)")
        L += ["", "## 검사", ""]
        if d["checks"]:
            L += ["| # | 검사 | 상태 | 내용 |", "|---|---|---|---|"]
            for i, c in enumerate(d["checks"]):
                L.append(f"| {i} | {_md_cell(c['name'])} | {c['status']} | {_md_cell(c['detail'])} |")
        else:
            L.append("(없음)")
        failed = [s for s in d["steps"] if not s.get("ok")]
        if failed:
            L += ["", "## 실패한 단계의 출력 끝부분"]
            for s in failed:
                L += ["", f"### {s.get('name')} (exit {s.get('exit_code')})", "", "```text"]
                L += [ln.replace("```", "'''") for ln in (s.get("output_tail") or [])]
                L.append("```")
        L.append("")
        return "\n".join(L)

    def write(self) -> tuple[Path, Path]:
        self.finished = now_iso()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stem = (re.sub(r"[^A-Za-z0-9_.-]+", "_", self.command).strip("_") or "nmfc") + "_report"
        jp, mp = self.out_dir / f"{stem}.json", self.out_dir / f"{stem}.md"
        d = self.to_dict()
        _write_text_atomic(jp, json.dumps(_jsonable(d), ensure_ascii=False, indent=2, default=_json_default) + "\n")
        _write_text_atomic(mp, self.to_markdown(d))
        return jp, mp


# ---------------------------------------------------------------------------
# 매니페스트 해시와 장치 정보
# ---------------------------------------------------------------------------
def manifest_sha256() -> str | None:
    """$NMFC_MANIFEST_SHA256, 없으면 REPO_ROOT/docker/context.lock의 원시 바이트 sha256, 그것도 없으면 None."""
    v = _env("NMFC_MANIFEST_SHA256")
    if v:
        return v.strip()
    lock = REPO_ROOT / "docker" / "context.lock"
    if lock.is_file():
        return sha256_file(lock)
    return None


def device_info() -> dict:
    """{torch, cuda, cudnn, gpu_name, capability, gpu_mem, device}. torch는 여기서만 import한다."""
    info: dict = {"torch": None, "cuda": None, "cudnn": None, "gpu_name": None, "capability": None,
                  "gpu_mem": None, "device": "cpu"}
    try:
        import torch  # noqa: PLC0415 (지연 import)
    except Exception as e:  # pragma: no cover - torch 없는 환경
        info["error"] = f"torch import 실패: {type(e).__name__}: {e}"
        return info
    info["torch"] = str(torch.__version__)
    info["cuda"] = torch.version.cuda
    try:
        info["cudnn"] = torch.backends.cudnn.version()
    except Exception:
        info["cudnn"] = None
    try:
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            info["gpu_name"] = props.name
            info["capability"] = [int(props.major), int(props.minor)]
            info["gpu_mem"] = int(props.total_memory)
            info["device"] = "cuda"
    except Exception as e:  # pragma: no cover
        info["error"] = f"CUDA 조회 실패: {type(e).__name__}: {e}"
    return info
