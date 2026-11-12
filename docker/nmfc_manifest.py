"""manifest.files·context.lock 생성·검증과 기준본 스냅샷.

이 모듈에서 정한 세부
- 파일 목록: nmfc_common.list_files(저장소 최상위의 SKIP_DIRS 디렉터리와 어느 깊이든 __pycache__·.pytest_cache
  제외)에서 docker/context.lock과 docker/manifest.files를 뺀 것이다. 경로는 저장소 상대 POSIX이고, 파이썬 문자열
  순서로 정렬한다. 하위 폴더의 data/·runs/·out/ 등은 빼지 않는다(그 아래 새 파일도 extra로 잡힌다).
- docker/manifest.files: 경로를 한 줄에 하나씩 쓴다. UTF-8, LF, 마지막 줄에도 LF를 붙인다.
- docker/context.lock: JSON Lines `{"path": .., "size": .., "sha256": ..}`. 경로 순서로 정렬하고 UTF-8·LF로 쓴다
  (ensure_ascii=False). size와 sha256은 모두 **LF 정규화 후** 값이다. 그래서 CRLF 작업트리(Windows)와
  LF 작업트리(컨테이너)에서 만든 lock이 바이트 단위로 같다. 같은 트리에서는 항상 같은 바이트가 나온다.
- verify_lock
  - lock의 경로를 기준으로 missing/changed를 낸다. changed는 정규화 sha256으로 판정한다.
  - root를 걸어 lock에 없는 파일을 extra로 낸다. 단, 최상위 SKIP_DIRS(runs/ 포함)와 어느 깊이든 __pycache__·
    .pytest_cache 아래, docker/context.lock, docker/manifest.files, root 안에 있는 lock 파일 자신,
    ignore_extra_under 접두 아래 파일은 빼고 센다.
  - lock 옆에 manifest.files가 있으면, 그 목록이 lock의 경로 목록과 같은지도 본다.
    다르면 그 파일을 kind="changed"로 낸다.
  - 결과는 (path, kind) 순으로 정렬한다. 손상된 lock(JSON 오류·중복 경로·필드 누락)이면 ValueError다.
- make_baseline: dest가 비어 있지 않은 디렉터리면 FileExistsError를 낸다. 다른 기준본과 섞이지 않게 하고,
  지울 권한이 없는 경로를 실수로 덮어쓰지 않게 하려는 것이다. 복사는 원본 바이트 그대로 한다(copy2).
- snapshot: 경로를 글자 그대로 본다(글롭을 펼치지 않는다). 없는 파일은
  {"mtime_ns": None, "sha256": None, "exists": False}다.
- run_manifest_cli
  - argv 맨 앞의 "manifest" 토큰은 있어도 되고 없어도 된다.
  - 종료코드: 0 = 불일치 0건, 1 = 불일치 있음, 2 = 전제조건 불충족(lock·root·ctx 없음, 손상된 lock,
    dest가 비어 있지 않음, 기준본 0개, 인자 오류). finalize는 2를 보고 빌드 중 lock 생성 폴백으로 갈 수 있다.
  - --verify의 기본 root: lock이 `<X>/docker/context.lock` 모양이면 X, 아니면 REPO_ROOT.
  - --write-lock은 마지막 줄에 `NMFC_MANIFEST_SHA256=<lock의 원시 바이트 sha256>`을 출력한다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import nmfc_common as common

LOCK_REL = "docker/context.lock"
FILES_REL = "docker/manifest.files"
GENERATED = frozenset({LOCK_REL, FILES_REL})
DEFAULT_IGNORE_EXTRA_UNDER = ("results/", "logs/")
_HEX = frozenset("0123456789abcdef")


# ---------------------------------------------------------------------------
# 파일 목록
# ---------------------------------------------------------------------------
def git_file_list(repo: Path) -> list[str]:
    """`git ls-files -co --exclude-standard -z` 결과(중복 제거·정렬). git 실패 시 RuntimeError."""
    r = subprocess.run(["git", "-c", "core.quotepath=false", "-C", os.fspath(repo), "--no-optional-locks",
                        "ls-files", "-co", "--exclude-standard", "-z"],
                       stdin=subprocess.DEVNULL, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"git ls-files 실패(exit {r.returncode}): {r.stderr.decode('utf-8', 'replace').strip()}")
    return sorted({p for p in r.stdout.decode("utf-8", "surrogateescape").split("\0") if p})


def context_files(root: Path) -> list[str]:
    """lock 대상 파일 목록(정렬). SKIP_DIRS 아래와 lock·manifest.files 자신은 뺀다."""
    return [p for p in common.list_files(root) if p not in GENERATED]


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def _files_text(paths: list[str]) -> str:
    return "".join(p + "\n" for p in paths)


# ---------------------------------------------------------------------------
# lock 작성·읽기·검증
# ---------------------------------------------------------------------------
def write_lock(ctx: Path) -> Path:
    """ctx를 걸어 docker/manifest.files와 docker/context.lock을 쓰고 lock 경로를 돌려준다."""
    ctx = Path(ctx)
    if not ctx.is_dir():
        raise FileNotFoundError(f"컨텍스트 디렉터리가 없다: {ctx}")
    paths = context_files(ctx)
    lines = []
    for rel in paths:
        if "\n" in rel or "\r" in rel:
            raise ValueError(f"줄바꿈이 든 경로는 manifest에 쓸 수 없다: {rel!r}")
        size, sha = common.norm_size_sha256(ctx / rel)
        lines.append(json.dumps({"path": rel, "size": size, "sha256": sha}, ensure_ascii=False) + "\n")
    ddir = ctx / "docker"
    ddir.mkdir(parents=True, exist_ok=True)
    _write_bytes_atomic(ctx / FILES_REL, _files_text(paths).encode("utf-8"))
    lock = ctx / LOCK_REL
    _write_bytes_atomic(lock, "".join(lines).encode("utf-8"))      # lock은 마지막에 쓴다(있으면 완결)
    return lock


def read_lock(lock: Path) -> dict[str, dict]:
    """lock을 {path: {"path","size","sha256"}}로 읽는다. 손상되면 ValueError."""
    lock = Path(lock)
    text = lock.read_bytes().decode("utf-8-sig")
    out: dict[str, dict] = {}
    for no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError as ex:
            raise ValueError(f"{lock}:{no}: JSON 오류: {ex}") from None
        if not isinstance(e, dict) or not isinstance(e.get("path"), str) or not isinstance(e.get("sha256"), str) \
                or not isinstance(e.get("size"), int):
            raise ValueError(f"{lock}:{no}: path/size/sha256 필드가 없거나 형식이 틀렸다")
        if len(e["sha256"]) != 64 or not set(e["sha256"]) <= _HEX:
            raise ValueError(f"{lock}:{no}: sha256 형식 오류")
        if e["path"] in out:
            raise ValueError(f"{lock}:{no}: 중복 경로 {e['path']}")
        out[e["path"]] = e
    return out


def _rel_under(path: Path, root: Path) -> str | None:
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return None


def verify_lock(root: Path, lock: Path, ignore_extra_under=DEFAULT_IGNORE_EXTRA_UNDER) -> list[dict]:
    """root를 lock과 대조한다. 불일치 [{path, kind: missing|changed|extra, expected, actual}]."""
    root, lock = Path(root), Path(lock)
    expected = read_lock(lock)
    findings: list[dict] = []
    for rel, e in expected.items():
        p = root / rel
        if not p.is_file():
            findings.append({"path": rel, "kind": "missing", "expected": e["sha256"], "actual": None})
            continue
        sha = common.norm_sha256(p)
        if sha != e["sha256"]:
            findings.append({"path": rel, "kind": "changed", "expected": e["sha256"], "actual": sha})

    excluded = set(GENERATED)
    lock_rel = _rel_under(lock, root)
    if lock_rel:
        excluded.add(lock_rel)
    prefixes = tuple(p if p.endswith("/") else p + "/" for p in (ignore_extra_under or ()) if p)
    for rel in common.list_files(root):
        if rel in expected or rel in excluded:
            continue
        if prefixes and rel.startswith(prefixes):
            continue
        findings.append({"path": rel, "kind": "extra", "expected": None, "actual": common.norm_sha256(root / rel)})

    mf = lock.parent / "manifest.files"
    if mf.is_file():
        want = sorted(expected)
        got = [ln for ln in mf.read_bytes().decode("utf-8-sig").splitlines() if ln.strip()]
        if got != want:
            findings.append({"path": _rel_under(mf, root) or mf.resolve().as_posix(), "kind": "changed",
                             "expected": hashlib.sha256(_files_text(want).encode("utf-8")).hexdigest(),
                             "actual": common.norm_sha256(mf)})
    findings.sort(key=lambda f: (f["path"], f["kind"]))
    return findings


# ---------------------------------------------------------------------------
# 기준본과 스냅샷
# ---------------------------------------------------------------------------
def is_baseline_path(rel: str) -> bool:
    """make_baseline 대상인가(저장소 상대 POSIX 경로)."""
    parts = rel.split("/")
    name = parts[-1]
    if parts[0] == "results" and len(parts) >= 2 and name.endswith((".json", ".md", ".csv")):
        return True                                             # results/**/*.json|*.md|*.csv
    if parts[0] == "logs" and len(parts) == 3 and name == "history.json":
        return True                                             # logs/*/history.json
    return rel == LOCK_REL                                      # docker/context.lock(있으면)


def baseline_files(root: Path) -> list[str]:
    root = Path(root)
    out: list[str] = []
    for sub in ("results", "logs"):
        d = root / sub
        if d.is_dir():
            out += [f"{sub}/{r}" for r in common.list_files(d, repo_root=False) if is_baseline_path(f"{sub}/{r}")]
    if (root / LOCK_REL).is_file():
        out.append(LOCK_REL)
    return sorted(set(out))


def make_baseline(root: Path, dest: Path) -> int:
    """root의 기준본 대상 파일을 상대 경로를 보존해 dest로 복사하고 파일 수를 돌려준다."""
    root, dest = Path(root), Path(dest)
    if dest.exists():
        if not dest.is_dir() or any(dest.iterdir()):
            raise FileExistsError(f"기준본 위치가 비어 있지 않다: {dest}")
    rels = baseline_files(root)
    dest.mkdir(parents=True, exist_ok=True)
    for rel in rels:
        d = dest / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, d)
    return len(rels)


def snapshot(root: Path, rel_paths: list[str]) -> dict[str, dict]:
    """{path: {"mtime_ns", "sha256", "exists"}}. sha256은 LF 정규화 값."""
    root = Path(root)
    out: dict[str, dict] = {}
    for rel in rel_paths:
        key = rel if isinstance(rel, str) else Path(rel).as_posix()
        p = root / key
        try:
            st = p.stat()
            if not stat.S_ISREG(st.st_mode):
                raise FileNotFoundError(key)
            out[key] = {"mtime_ns": st.st_mtime_ns, "sha256": common.norm_sha256(p), "exists": True}
        except OSError:
            out[key] = {"mtime_ns": None, "sha256": None, "exists": False}
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _default_root_for(lock: Path) -> Path:
    lp = Path(lock).resolve()
    if lp.parent.name == "docker":
        return lp.parent.parent
    return common.REPO_ROOT


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="nmfc manifest",
        description="context.lock·manifest.files 생성(--write-lock), 검증(--verify), 기준본 스냅샷(--baseline).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write-lock", metavar="CTX", help="CTX를 걸어 CTX/docker/{context.lock,manifest.files}를 쓴다")
    g.add_argument("--verify", metavar="LOCK", help="root를 LOCK과 대조한다(불일치 0건이면 exit 0)")
    g.add_argument("--baseline", metavar="DEST", help="root의 기준본 대상 파일을 DEST로 복사한다")
    ap.add_argument("--root", metavar="R", help="--verify/--baseline의 대상 트리. 기본: --verify는 LOCK이 "
                                                "<X>/docker/context.lock이면 X, --baseline은 저장소 루트")
    ap.add_argument("--strict", action="store_true",
                    help="--verify에서 results/·logs/ 아래 새 파일도 extra로 센다(빌드 중 검사용)")
    ap.add_argument("--json", metavar="PATH", help="--verify 결과를 JSON으로도 저장한다")
    return ap


def run_manifest_cli(argv: list[str]) -> int:
    """`manifest --write-lock CTX | --verify LOCK [--root R] [--strict] | --baseline DEST`. 0/1/2를 돌려준다."""
    argv = [os.fspath(a) for a in (argv or [])]
    if argv and argv[0] == "manifest":
        argv = argv[1:]
    try:
        a = _parser().parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 2

    if a.write_lock:
        ctx = Path(a.write_lock)
        if not ctx.is_dir():
            print(f"[manifest] 전제조건 불충족: 컨텍스트 디렉터리가 없다: {ctx}", file=sys.stderr)
            return 2
        lock = write_lock(ctx)
        n = len(read_lock(lock))
        print(f"[manifest] manifest.files 작성: {ctx / FILES_REL}")
        print(f"[manifest] context.lock 작성: {lock} ({n}개 파일)")
        print(f"NMFC_MANIFEST_SHA256={common.sha256_file(lock)}")
        return 0

    if a.verify:
        lock = Path(a.verify)
        if not lock.is_file():
            print(f"[manifest] 전제조건 불충족: lock이 없다: {lock}", file=sys.stderr)
            return 2
        root = Path(a.root) if a.root else _default_root_for(lock)
        if not root.is_dir():
            print(f"[manifest] 전제조건 불충족: root가 없다: {root}", file=sys.stderr)
            return 2
        try:
            findings = verify_lock(root, lock, () if a.strict else DEFAULT_IGNORE_EXTRA_UNDER)
        except ValueError as e:
            print(f"[manifest] 전제조건 불충족: 손상된 lock: {e}", file=sys.stderr)
            return 2
        if a.json:
            jp = Path(a.json)
            jp.parent.mkdir(parents=True, exist_ok=True)
            with open(jp, "w", encoding="utf-8", newline="\n") as fh:
                json.dump({"root": root.resolve().as_posix(), "lock": lock.resolve().as_posix(),
                           "strict": bool(a.strict), "findings": findings}, fh, ensure_ascii=False, indent=2)
                fh.write("\n")
        n = len(read_lock(lock))
        if not findings:
            print(f"[manifest] OK: {n}개 파일이 lock과 일치 (root={root}, strict={bool(a.strict)})")
            return 0
        kinds = {k: sum(f["kind"] == k for f in findings) for k in ("missing", "changed", "extra")}
        print(f"[manifest] FAIL: 불일치 {len(findings)}건 {kinds} (root={root}, strict={bool(a.strict)})")
        for f in findings[:200]:
            print(f"  {f['kind']:<8} {f['path']}")
        if len(findings) > 200:
            print(f"  ... 외 {len(findings) - 200}건")
        return 1

    dest = Path(a.baseline)
    root = Path(a.root) if a.root else common.REPO_ROOT
    if not root.is_dir():
        print(f"[manifest] 전제조건 불충족: root가 없다: {root}", file=sys.stderr)
        return 2
    try:
        n = make_baseline(root, dest)
    except FileExistsError as e:
        print(f"[manifest] 전제조건 불충족: {e}", file=sys.stderr)
        return 2
    if n == 0:
        print(f"[manifest] 전제조건 불충족: 기준본 대상 파일이 0개다(root={root})", file=sys.stderr)
        return 2
    print(f"[manifest] 기준본 {n}개 파일 → {dest}")
    return 0


if __name__ == "__main__":
    common.setup_stdio()
    sys.exit(run_manifest_cli(sys.argv[1:]))
