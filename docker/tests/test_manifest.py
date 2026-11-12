"""nmfc_manifest 테스트: lock 작성(형식·결정성·CRLF 불변), 검증(망가뜨린 입력 → changed/missing/extra), 기준본, 스냅샷, CLI."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import nmfc_common as C
import nmfc_manifest as M

DOCKER_DIR = Path(M.__file__).resolve().parent
PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) + b"\r\n" * 8

TREE = {
    "README.md": "# NMFC\n두 번째 줄\n".encode("utf-8"),
    ".gitattributes": b"* text=auto eol=lf\n*.png binary\n",
    "src/a.py": b"x = 1\n",
    "notes/\ud55c\uae00\ud45c.csv": "\ufeff실험,값\n1,2\n".encode("utf-8"),   # notes/한글표.csv (BOM)
    "docker/expected.yaml": "# 기대값\n".encode("utf-8"),
    "results/stats_ci.json": b'{"map": 0.5}\n',
    "results/notes.md": b"# n\n",
    "results/openset_noise_grid/master.csv": b"a,b\n1,2\n",
    "results/openset_noise_grid/curves_auc.png": PNG,
    "results/c100v_shrb/metrics.json": b'{"top1": 0.6}\n',
    "results/c100v_shrb/embeddings.npz": b"PK\x03\x04" + bytes(100),
    "logs/c100v_shrb/history.json": b"[]\n",
}
SKIPPED = {
    ".git/HEAD": b"ref: refs/heads/main\n",
    "data/raw/test_32x32.mat": b"\x00mat",
    "checkpoints/c.pt": b"\x00",
    "out/verify_report.json": b"{}",
    "dist/nmfc-cpu.tar.gz": b"\x1f\x8b",
    "runs/pipeline_status.json": b"{}",
    "_planning/PROGRESS.md": b"p",
    "src/__pycache__/a.cpython-314.pyc": b"\x00",
    ".pytest_cache/v/cache": b"c",
}


def _make(root: Path, files: dict) -> Path:
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return root


@pytest.fixture
def ctx(tmp_path) -> Path:
    return _make(tmp_path / "ctx", {**TREE, **SKIPPED})


def _to_crlf(root: Path) -> None:
    for rel in C.list_files(root):
        p = root / rel
        data = p.read_bytes()
        if not C.is_binary(p, data):
            p.write_bytes(data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))


# ---------------------------------------------------------------------------
# write_lock
# ---------------------------------------------------------------------------
def test_write_lock_format_and_exclusions(ctx):
    lock = M.write_lock(ctx)
    assert lock == ctx / "docker" / "context.lock"
    files = (ctx / "docker" / "manifest.files").read_bytes()
    assert b"\r\n" not in files and files.endswith(b"\n")
    listed = files.decode("utf-8").splitlines()
    assert listed == sorted(TREE)                                       # SKIP_DIRS와 lock 자신은 없다
    raw = lock.read_bytes()
    assert b"\r\n" not in raw and raw.endswith(b"\n")
    assert "notes/한글표.csv".encode("utf-8") in raw                     # UTF-8 그대로(\\u 이스케이프 아님)
    entries = [json.loads(ln) for ln in raw.decode("utf-8").splitlines()]
    assert [e["path"] for e in entries] == sorted(TREE)
    for e in entries:
        assert list(e) == ["path", "size", "sha256"]
        norm = C.norm_bytes(e["path"], TREE[e["path"]])
        assert e["sha256"] == hashlib.sha256(norm).hexdigest() == C.norm_sha256(ctx / e["path"])
        assert e["size"] == len(norm)


def test_write_lock_is_deterministic_and_crlf_invariant(ctx, tmp_path):
    lock1 = M.write_lock(ctx).read_bytes()
    lock2 = M.write_lock(ctx).read_bytes()                              # 두 번째 실행에서 lock 자신이 섞이지 않는다
    assert lock1 == lock2
    crlf = tmp_path / "crlf"
    shutil.copytree(ctx, crlf)
    for rel in ("docker/context.lock", "docker/manifest.files"):
        (crlf / rel).unlink()
    _to_crlf(crlf)
    assert b"\r\n" in (crlf / "README.md").read_bytes()
    assert M.write_lock(crlf).read_bytes() == lock1                     # Windows(CRLF)와 컨테이너(LF)에서 같은 lock
    assert (crlf / "docker/manifest.files").read_bytes() == (ctx / "docker/manifest.files").read_bytes()


def test_write_lock_under_cp949_parent_writes_utf8_lf(ctx, tmp_path):
    env = dict(os.environ)
    env["PYTHONUTF8"] = "0"
    env.pop("PYTHONIOENCODING", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    p = subprocess.run([sys.executable, "-B", str(DOCKER_DIR / "nmfc_manifest.py"), "--write-lock", str(ctx)],
                       env=env, capture_output=True, encoding="utf-8", errors="replace", timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    raw = (ctx / "docker" / "context.lock").read_bytes()
    assert p.stdout.strip().splitlines()[-1] == "NMFC_MANIFEST_SHA256=" + hashlib.sha256(raw).hexdigest()
    raw.decode("utf-8")                                                 # cp949로 쓰였다면 여기서 실패
    assert b"\r\n" not in raw and "notes/한글표.csv".encode("utf-8") in raw
    assert "notes/한글표.csv" in (ctx / "docker" / "manifest.files").read_bytes().decode("utf-8").splitlines()
    assert M.verify_lock(ctx, ctx / "docker" / "context.lock") == []
    same = _make(tmp_path / "same", {**TREE, **SKIPPED})                # 같은 트리를 이 프로세스(UTF-8 모드)에서
    assert M.write_lock(same).read_bytes() == raw


# ---------------------------------------------------------------------------
# verify_lock — 깨끗한 트리는 통과, 망가뜨린 트리는 검출
# ---------------------------------------------------------------------------
def test_verify_lock_clean_tree_passes(ctx):
    lock = M.write_lock(ctx)
    assert M.verify_lock(ctx, lock) == []
    assert M.verify_lock(ctx, lock, ignore_extra_under=()) == []        # 엄격 모드도 통과


def test_verify_lock_detects_one_byte_change(ctx):
    lock = M.write_lock(ctx)
    p = ctx / "src" / "a.py"
    before = M.read_lock(lock)["src/a.py"]["sha256"]
    p.write_bytes(b"x = 2\n")                                           # 내용 1바이트 변경(같은 크기)
    f = M.verify_lock(ctx, lock)
    assert f == [{"path": "src/a.py", "kind": "changed", "expected": before, "actual": C.norm_sha256(p)}]


def test_verify_lock_detects_change_under_ignored_prefix(ctx):
    lock = M.write_lock(ctx)
    (ctx / "results" / "stats_ci.json").write_bytes(b'{"map": 0.6}\n')  # ignore_extra_under는 extra에만 적용
    f = M.verify_lock(ctx, lock)
    assert [(x["path"], x["kind"]) for x in f] == [("results/stats_ci.json", "changed")]


def test_verify_lock_crlf_only_rewrite_is_not_flagged(ctx):
    lock = M.write_lock(ctx)
    _to_crlf(ctx)                                                       # Windows 텍스트 모드 재기록 흉내
    assert b"\r\n" in (ctx / "results" / "stats_ci.json").read_bytes()
    assert M.verify_lock(ctx, lock) == []


def test_verify_lock_binary_crlf_flip_is_flagged(ctx):
    lock = M.write_lock(ctx)
    p = ctx / "results" / "openset_noise_grid" / "curves_auc.png"
    p.write_bytes(PNG.replace(b"\r\n", b"\n"))                          # 바이너리는 정규화하지 않는다
    assert [(x["path"], x["kind"]) for x in M.verify_lock(ctx, lock)] == [
        ("results/openset_noise_grid/curves_auc.png", "changed")]


def test_verify_lock_detects_missing(ctx):
    lock = M.write_lock(ctx)
    expected = M.read_lock(lock)["notes/한글표.csv"]["sha256"]
    (ctx / "notes" / "한글표.csv").unlink()
    assert M.verify_lock(ctx, lock) == [
        {"path": "notes/한글표.csv", "kind": "missing", "expected": expected, "actual": None}]


def test_verify_lock_detects_extra_outside_ignored_prefixes(ctx):
    lock = M.write_lock(ctx)
    (ctx / "src" / "new.py").write_bytes(b"y = 1\n")
    f = M.verify_lock(ctx, lock)
    assert f == [{"path": "src/new.py", "kind": "extra", "expected": None,
                  "actual": hashlib.sha256(b"y = 1\n").hexdigest()}]


def test_verify_lock_extra_under_results_only_in_strict_mode(ctx):
    lock = M.write_lock(ctx)
    (ctx / "results" / "new_eval.json").write_bytes(b"{}\n")
    (ctx / "logs" / "x" / "history.json").parent.mkdir(parents=True)
    (ctx / "logs" / "x" / "history.json").write_bytes(b"[]\n")
    assert M.verify_lock(ctx, lock) == []                               # 기본: results/·logs/ 아래 새 파일 무시
    assert M.DEFAULT_IGNORE_EXTRA_UNDER == ("results/", "logs/")
    strict = M.verify_lock(ctx, lock, ignore_extra_under=())
    assert [(x["path"], x["kind"]) for x in strict] == [("logs/x/history.json", "extra"),
                                                       ("results/new_eval.json", "extra")]
    only_results = M.verify_lock(ctx, lock, ignore_extra_under=("results",))   # 슬래시 없이 줘도 디렉터리 접두
    assert [x["path"] for x in only_results] == ["logs/x/history.json"]


def test_verify_lock_never_counts_skip_dirs_as_extra(ctx):
    lock = M.write_lock(ctx)
    for rel in ("runs/new_status.json", "out/regen_report.json", "src/__pycache__/b.pyc", "data/raw/new.bin",
                "checkpoints/new.pt", ".git/index", "configs/__pycache__/z.pyc", "_planning/x.txt"):
        (ctx / rel).parent.mkdir(parents=True, exist_ok=True)
        (ctx / rel).write_bytes(b"\x00new")
    assert M.verify_lock(ctx, lock, ignore_extra_under=()) == []


NESTED_SKIP_NAMES = ("results/data/evil.json", "logs/x/out/evil.png", "src/runs/evil.py",
                     "scripts/checkpoints/evil.py", "results/c100v_shrb/dist/metrics.json", "configs/_planning/n.md")


def test_skip_names_below_top_level_are_not_skipped(ctx):
    """SKIP_DIRS 이름(data·out·runs 등)은 저장소 최상위에서만 뺀다. 하위 폴더에 생긴 파일은 lock 대조에서 extra로
    잡혀야 한다(깊이 무관 제외 때문에 results/data/·src/runs/ 등의 새 파일이 '예상 밖 변경 0건'을 통과했다)."""
    lock = M.write_lock(ctx)
    for rel in NESTED_SKIP_NAMES:
        (ctx / rel).parent.mkdir(parents=True, exist_ok=True)
        (ctx / rel).write_bytes(b"{}")
    (ctx / "results" / "__pycache__").mkdir()
    (ctx / "results" / "__pycache__" / "x.pyc").write_bytes(b"\x00")          # 깊이 무관 이름은 여전히 뺀다
    (ctx / "data" / "raw" / "more.bin").write_bytes(b"\x00")                    # 최상위 data/는 여전히 뺀다
    found = M.verify_lock(ctx, lock, ignore_extra_under=())
    assert sorted(f["path"] for f in found) == sorted(NESTED_SKIP_NAMES)
    assert {f["kind"] for f in found} == {"extra"}
    assert M.run_manifest_cli(["--verify", str(lock), "--root", str(ctx), "--strict"]) == 1
    relock = M.write_lock(ctx)                                                  # 새로 쓴 lock에는 들어간다
    listed = set(M.read_lock(relock))
    assert set(NESTED_SKIP_NAMES) <= listed and "results/__pycache__/x.pyc" not in listed
    assert not any(p.startswith(("data/", "runs/", "out/", "_planning/", ".git/")) for p in listed)


def test_make_baseline_includes_nested_skip_names(ctx, tmp_path):
    _make(ctx, {"results/data/deep.json": b"{}", "results/x/out/r.md": b"# r\n"})
    dest = tmp_path / "bl"
    M.make_baseline(ctx, dest)
    assert (dest / "results" / "data" / "deep.json").is_file() and (dest / "results" / "x" / "out" / "r.md").is_file()


def test_verify_lock_checks_manifest_files_consistency(ctx):
    lock = M.write_lock(ctx)
    mf = ctx / "docker" / "manifest.files"
    lines = mf.read_bytes().decode("utf-8").splitlines()
    mf.write_bytes(("\n".join(lines[1:]) + "\n").encode("utf-8"))      # 한 줄 빼기
    assert [(x["path"], x["kind"]) for x in M.verify_lock(ctx, lock)] == [("docker/manifest.files", "changed")]
    mf.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))      # CRLF로만 바뀐 것은 일치로 본다
    assert M.verify_lock(ctx, lock) == []


def test_verify_lock_with_lock_outside_root(ctx, tmp_path):
    M.write_lock(ctx)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    lock = elsewhere / "context.lock"
    shutil.copyfile(ctx / "docker" / "context.lock", lock)
    assert M.verify_lock(ctx, lock) == []
    (ctx / "README.md").write_bytes(b"# changed\n")
    assert [x["kind"] for x in M.verify_lock(ctx, lock)] == ["changed"]


def test_read_lock_rejects_corrupt_lock(tmp_path):
    good = json.dumps({"path": "a", "size": 1, "sha256": "0" * 64})
    cases = {
        "badjson": good + "\n{not json\n",
        "dup": good + "\n" + good + "\n",
        "nofield": json.dumps({"path": "a", "sha256": "0" * 64}) + "\n",
        "badsha": json.dumps({"path": "a", "size": 1, "sha256": "xyz"}) + "\n",
    }
    for name, text in cases.items():
        p = tmp_path / f"{name}.lock"
        p.write_bytes(text.encode("utf-8"))
        with pytest.raises(ValueError):
            M.read_lock(p)
        assert M.run_manifest_cli(["--verify", str(p), "--root", str(tmp_path)]) == 2


# ---------------------------------------------------------------------------
# make_baseline / snapshot
# ---------------------------------------------------------------------------
def test_make_baseline_copies_exact_target_set(ctx, tmp_path):
    _make(ctx, {
        "results/top.csv": b"a\r\nb\r\n",                               # CRLF 그대로 복사(정규화는 비교 단계 몫)
        "results/deep/a/b/c.json": b"{}",
        "results/fig.png": PNG,                                         # results PNG는 대상 아님
        "results/x.txt": b"t",
        "logs/history.json": b"[]",                                     # logs/*/history.json만
        "logs/e1/sub/history.json": b"[]",
        "logs/e1/other.json": b"{}",
        "logs/e1/train.log": b"log",
    })
    M.write_lock(ctx)
    dest = tmp_path / "baseline"
    n = M.make_baseline(ctx, dest)
    want = sorted([
        "results/stats_ci.json", "results/notes.md", "results/openset_noise_grid/master.csv",
        "results/c100v_shrb/metrics.json", "results/top.csv", "results/deep/a/b/c.json",
        "logs/c100v_shrb/history.json",
        "docker/context.lock",
    ])
    assert n == len(want)
    assert C.list_files(dest) == want
    for rel in want:
        assert (dest / rel).read_bytes() == (ctx / rel).read_bytes(), rel
    with pytest.raises(FileExistsError):                                # 비어 있지 않은 dest는 거부
        M.make_baseline(ctx, dest)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert M.make_baseline(ctx, empty) == n                             # 빈 디렉터리는 허용


def test_make_baseline_without_lock(ctx, tmp_path):
    n = M.make_baseline(ctx, tmp_path / "b")
    assert "docker/context.lock" not in C.list_files(tmp_path / "b")
    assert n == len(C.list_files(tmp_path / "b")) == 5


def test_snapshot_records_mtime_hash_existence(ctx):
    rels = ["results/stats_ci.json", "results/openset_noise_grid/curves_auc.png", "results/nope.json", "results"]
    snap = M.snapshot(ctx, rels)
    assert list(snap) == rels
    p = ctx / "results" / "stats_ci.json"
    assert snap["results/stats_ci.json"] == {"mtime_ns": p.stat().st_mtime_ns, "sha256": C.norm_sha256(p),
                                             "exists": True}
    assert snap["results/nope.json"] == {"mtime_ns": None, "sha256": None, "exists": False}
    assert snap["results"]["exists"] is False                           # 디렉터리는 파일이 아니다
    p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))               # 같은 내용 CRLF 재기록
    os.utime(p, ns=(p.stat().st_atime_ns, snap["results/stats_ci.json"]["mtime_ns"] + 10**9))
    snap2 = M.snapshot(ctx, ["results/stats_ci.json"])
    assert snap2["results/stats_ci.json"]["mtime_ns"] != snap["results/stats_ci.json"]["mtime_ns"]
    assert snap2["results/stats_ci.json"]["sha256"] == snap["results/stats_ci.json"]["sha256"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_manifest_cli_roundtrip(ctx, tmp_path, capsys):
    assert M.run_manifest_cli(["--write-lock", str(ctx)]) == 0
    lock = ctx / "docker" / "context.lock"
    out = capsys.readouterr().out
    assert out.strip().splitlines()[-1] == "NMFC_MANIFEST_SHA256=" + hashlib.sha256(lock.read_bytes()).hexdigest()

    # 맨 앞 'manifest' 토큰 허용, root는 <X>/docker/context.lock → X로 유도
    assert M.run_manifest_cli(["manifest", "--verify", str(lock)]) == 0
    assert M.run_manifest_cli(["--verify", str(lock), "--strict"]) == 0

    (ctx / "results" / "new.json").write_bytes(b"{}")
    assert M.run_manifest_cli(["--verify", str(lock)]) == 0
    assert M.run_manifest_cli(["--verify", str(lock), "--strict"]) == 1

    (ctx / "src" / "a.py").write_bytes(b"x = 9\n")
    rj = tmp_path / "rep" / "manifest_verify.json"
    assert M.run_manifest_cli(["--verify", str(lock), "--root", str(ctx), "--json", str(rj)]) == 1
    rep = json.loads(rj.read_text(encoding="utf-8"))
    assert [(f["path"], f["kind"]) for f in rep["findings"]] == [("src/a.py", "changed")]
    assert "FAIL" in capsys.readouterr().out

    assert M.run_manifest_cli(["--verify", str(tmp_path / "no.lock")]) == 2
    assert M.run_manifest_cli(["--verify", str(lock), "--root", str(tmp_path / "noroot")]) == 2
    assert M.run_manifest_cli(["--write-lock", str(tmp_path / "noctx")]) == 2

    dest = tmp_path / "bl"
    assert M.run_manifest_cli(["--baseline", str(dest), "--root", str(ctx)]) == 0
    assert (dest / "docker" / "context.lock").is_file() and (dest / "results" / "stats_ci.json").is_file()
    assert M.run_manifest_cli(["--baseline", str(dest), "--root", str(ctx)]) == 2       # 비어 있지 않음
    bare = tmp_path / "bare"
    bare.mkdir()
    assert M.run_manifest_cli(["--baseline", str(tmp_path / "bl2"), "--root", str(bare)]) == 2   # 0개

    assert M.run_manifest_cli([]) == 2                                  # 인자 오류
    assert M.run_manifest_cli(["--verify", str(lock), "--baseline", str(dest)]) == 2
    assert M.run_manifest_cli(["--help"]) == 0


# ---------------------------------------------------------------------------
# git_file_list
# ---------------------------------------------------------------------------
def _git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, encoding="utf-8", errors="replace",
                          stdin=subprocess.DEVNULL)


@pytest.mark.skipif(shutil.which("git") is None, reason="git 없음")
def test_git_file_list_temp_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    assert _git("-c", "init.defaultBranch=main", "init", "-q", cwd=repo).returncode == 0
    _make(repo, {".gitignore": b"*.log\nbuild/\n", "a.txt": b"a", "sub/b.txt": b"b", "x.log": b"l",
                 "build/o.txt": b"o", "한글.txt": "가".encode("utf-8")})
    assert M.git_file_list(repo) == [".gitignore", "a.txt", "sub/b.txt", "한글.txt"]
    with pytest.raises(RuntimeError):
        M.git_file_list(tmp_path / "not-a-repo-dir")


@pytest.mark.skipif(shutil.which("git") is None or not (C.REPO_ROOT / ".git").exists(),
                    reason="git 저장소가 아님(컨테이너 등)")
def test_git_file_list_real_repo_matches_walker():
    """실제 저장소(읽기 전용): git 목록의 모든 현존 파일이 list_files(=lock 대상)에 있어야 한다.

    SKIP_DIRS 이름을 가진 하위 디렉터리에 추적 파일이 생기면 lock에서 조용히 빠지는데, 이것을 잡는다.
    """
    g = M.git_file_list(C.REPO_ROOT)
    w = set(C.list_files(C.REPO_ROOT))
    assert g == sorted(set(g))
    assert "src/utils/misc.py" in g and "docker/nmfc_cli.py" in g
    assert not any(p.startswith("_planning/") for p in g)
    missing = [p for p in g if (C.REPO_ROOT / p).is_file() and p not in w]
    assert missing == []
