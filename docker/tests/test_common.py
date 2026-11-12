"""nmfc_common 테스트: LF 정규화·바이너리 판정, 환경, 작업 사본, run_step(expect·exit·timeout·cp949 부모), 보고서."""
import codecs
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

import nmfc_common as C

DOCKER_DIR = Path(C.__file__).resolve().parent


def _script(d: Path, name: str, body: str) -> Path:
    p = d / name
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def _plain_env() -> dict:
    """테스트가 띄우는 '부모' 프로세스용 환경: UTF-8 모드를 끈다(cp949 부모 재현)."""
    env = dict(os.environ)
    env["PYTHONUTF8"] = "0"
    env.pop("PYTHONIOENCODING", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


# ---------------------------------------------------------------------------
# LF 정규화와 바이너리 판정
# ---------------------------------------------------------------------------
def test_norm_bytes_text_crlf_to_lf_and_keeps_lone_cr():
    assert C.norm_bytes("a.txt", b"a\r\nb\r\n") == b"a\nb\n"
    assert C.norm_bytes(Path("x/y.py"), b"x = 1\r\n") == b"x = 1\n"
    assert C.norm_bytes("a.txt", b"a\rb\r\r\n") == b"a\rb\r\n"          # 단독 CR 유지, CRLF만 LF
    assert C.norm_bytes(".gitignore", b"out/\r\n") == b"out/\n"           # 확장자 없는 점 파일도 텍스트


def test_is_binary_by_extension_case_insensitive():
    for name in ("a.png", "A.PNG", "x.tar.gz", "m.pt", "w.pth", "d.npz", "e.npy", "f.pdf", "g.pptx",
                 "h.zip", "i.xlsx", "j.mat", "k.jpg", "l.JPEG"):
        assert C.is_binary(name, b"plain text\r\n"), name
        assert C.norm_bytes(name, b"plain\r\n") == b"plain\r\n", name
    assert not C.is_binary("a.json", b"{}\r\n")
    assert not C.is_binary("a.csv", "가,나\r\n".encode("utf-8"))


def test_is_binary_by_nul_only_in_first_8000_bytes():
    head_nul = b"abc\x00def\r\n"
    assert C.is_binary("a.txt", head_nul)
    assert C.norm_bytes("a.txt", head_nul) == head_nul
    at_7999 = b"a" * 7999 + b"\x00" + b"\r\n"
    assert C.is_binary("a.txt", at_7999)
    at_8000 = b"a" * 8000 + b"\x00" + b"\r\n"                          # 8000번째 이후 NUL → 텍스트
    assert not C.is_binary("a.txt", at_8000)
    assert C.norm_bytes("a.txt", at_8000) == b"a" * 8000 + b"\x00\n"


def test_norm_sha256_crlf_equals_lf(tmp_path):
    lf = "첫 줄\n둘째 줄\n".encode("utf-8")
    (tmp_path / "lf.md").write_bytes(lf)
    (tmp_path / "crlf.md").write_bytes(lf.replace(b"\n", b"\r\n"))
    want = hashlib.sha256(lf).hexdigest()
    assert C.norm_sha256(tmp_path / "lf.md") == want
    assert C.norm_sha256(tmp_path / "crlf.md") == want
    assert C.norm_size_sha256(tmp_path / "crlf.md") == (len(lf), want)
    assert C.sha256_file(tmp_path / "crlf.md") != want                   # 원시 해시는 다르다


def test_norm_sha256_binary_is_not_normalized(tmp_path):
    png = b"\x89PNG\r\n\x1a\n" + b"\r\n" * 10
    (tmp_path / "a.png").write_bytes(png)
    (tmp_path / "b.png").write_bytes(png.replace(b"\r\n", b"\n"))
    assert C.norm_sha256(tmp_path / "a.png") == hashlib.sha256(png).hexdigest()
    assert C.norm_sha256(tmp_path / "a.png") != C.norm_sha256(tmp_path / "b.png")


def test_norm_sha256_large_files_across_chunk_boundary(tmp_path):
    rng = np.random.default_rng(0)
    blob = rng.integers(0, 256, size=3 * (1 << 20) + 17, dtype=np.uint8).tobytes()
    (tmp_path / "big.npz").write_bytes(blob)
    assert C.norm_sha256(tmp_path / "big.npz") == hashlib.sha256(blob).hexdigest()
    # 텍스트: CRLF가 1 MiB 청크 경계에 걸치게 만든다(\r이 첫 청크 끝, \n이 둘째 청크 처음)
    first = b"x" * ((1 << 20) - 1) + b"\r"
    text = first + b"\n" + b"y\r\n" * 1000
    (tmp_path / "big.txt").write_bytes(text)
    assert C.norm_sha256(tmp_path / "big.txt") == hashlib.sha256(text.replace(b"\r\n", b"\n")).hexdigest()


# ---------------------------------------------------------------------------
# 환경
# ---------------------------------------------------------------------------
def test_env_helpers_read_environment_at_call_time(monkeypatch, tmp_path):
    monkeypatch.delenv("NMFC_IN_CONTAINER", raising=False)
    assert not C.in_container()
    monkeypatch.setenv("NMFC_IN_CONTAINER", "1")
    assert C.in_container()
    monkeypatch.setenv("NMFC_IN_CONTAINER", "0")
    assert not C.in_container()

    monkeypatch.delenv("NMFC_OUT", raising=False)
    assert C.default_out() == C.REPO_ROOT / "out"
    monkeypatch.setenv("NMFC_OUT", "")
    assert C.default_out() == C.REPO_ROOT / "out"
    monkeypatch.setenv("NMFC_OUT", str(tmp_path / "o"))
    assert C.default_out() == tmp_path / "o"

    monkeypatch.delenv("NMFC_BASELINE", raising=False)
    assert C.default_baseline() is None
    monkeypatch.setenv("NMFC_BASELINE", str(tmp_path / "b"))
    assert C.default_baseline() == tmp_path / "b"


def test_manifest_sha256_env_then_lock(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "REPO_ROOT", tmp_path)
    monkeypatch.delenv("NMFC_MANIFEST_SHA256", raising=False)
    assert C.manifest_sha256() is None
    (tmp_path / "docker").mkdir()
    lock = tmp_path / "docker" / "context.lock"
    lock.write_bytes(b'{"path": "a", "size": 1, "sha256": "00"}\n')
    assert C.manifest_sha256() == hashlib.sha256(lock.read_bytes()).hexdigest()
    monkeypatch.setenv("NMFC_MANIFEST_SHA256", "")                     # 빈 build-arg는 없는 것으로
    assert C.manifest_sha256() == hashlib.sha256(lock.read_bytes()).hexdigest()
    monkeypatch.setenv("NMFC_MANIFEST_SHA256", "abc123")
    assert C.manifest_sha256() == "abc123"


def test_child_env_merges_and_forces_utf8(monkeypatch):
    monkeypatch.setenv("PYTHONUTF8", "0")
    monkeypatch.setenv("NMFC_TEST_DROP", "x")
    env = C.child_env({"CUDA_VISIBLE_DEVICES": -1, "NMFC_TEST_DROP": None, "PYTHONUTF8": "0"})
    assert env["CUDA_VISIBLE_DEVICES"] == "-1"
    assert "NMFC_TEST_DROP" not in env
    assert env["PYTHONUTF8"] == "1"                                     # 강제
    assert env["PYTHONDONTWRITEBYTECODE"] == "1" and env["PYTHONUNBUFFERED"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert any(k.upper() == "PATH" for k in env)                        # os.environ 위에 덧씌운다


# ---------------------------------------------------------------------------
# 작업 사본
# ---------------------------------------------------------------------------
def _fake_repo(root: Path) -> None:
    files = {
        "README.md": b"# r\n",
        "src/a.py": b"x = 1\n",
        "src/__pycache__/a.cpython-314.pyc": b"\x00pyc",
        "results/r.json": b"{}\n",
        "notes/한글.csv": "가,나\n".encode("utf-8"),
        "data/raw/big.bin": b"\x00" * 10,
        "checkpoints/c.pt": b"\x00",
        "out/x.json": b"{}",
        "dist/x.tar.gz": b"\x00",
        "runs/pipeline_status.json": b"{}",
        "_planning/n.txt": b"n",
        ".git/HEAD": b"ref: refs/heads/main\n",
        ".pytest_cache/v": b"v",
        "notes/out": b"file named like a skip dir",                     # 파일 이름은 건너뛰지 않는다
    }
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    (root / "out_file").write_bytes(b"k")                               # 'out'과 비슷한 이름의 파일
    (root / "dist2").mkdir()
    (root / "dist2" / "keep.txt").write_bytes(b"k")


def test_make_workdir_host_copies_without_skip_dirs(monkeypatch, tmp_path):
    monkeypatch.delenv("NMFC_IN_CONTAINER", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    src = tmp_path / "src"
    _fake_repo(src)
    wd = C.make_workdir(src)
    try:
        assert wd == tmp_path / "tmp" / f"nmfc-work-{os.getpid()}"
        got = C.list_files(wd)
        assert got == ["README.md", "dist2/keep.txt", "notes/out", "notes/한글.csv", "out_file",
                       "results/r.json", "src/a.py"]
        for d in C.SKIP_DIRS:
            assert not (wd / d).exists(), d
        assert not (wd / "src" / "__pycache__").exists()
        assert (wd / "notes" / "한글.csv").read_bytes() == "가,나\n".encode("utf-8")
        assert (src / "data" / "raw" / "big.bin").exists()                # 원본은 그대로

        (wd / "stale.txt").write_text("old", encoding="utf-8")         # 이미 있으면 지우고 새로 만든다
        wd2 = C.make_workdir(src)
        assert wd2 == wd and not (wd / "stale.txt").exists()
    finally:
        C.cleanup_workdir(wd, keep=False)
    assert not wd.exists()


def test_make_workdir_copies_nested_skip_names_but_not_top_level(monkeypatch, tmp_path):
    """최상위 data/·runs/ 등은 복사하지 않지만, 하위 폴더의 같은 이름(src/data/, results/x/out/)은 복사한다.
    __pycache__·.pytest_cache는 어느 깊이에서나 뺀다."""
    monkeypatch.delenv("NMFC_IN_CONTAINER", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    src = tmp_path / "src"
    _fake_repo(src)
    for rel in ("src/data/loader.py", "results/x/out/r.json", "configs/runs/n.yaml", "src/pkg/.pytest_cache/v"):
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_bytes(b"k")
    wd = C.make_workdir(src)
    try:
        got = set(C.list_files(wd))
        assert {"src/data/loader.py", "results/x/out/r.json", "configs/runs/n.yaml"} <= got
        assert not any(p.startswith(("data/", "runs/", "out/", "dist/", "checkpoints/", "_planning/", ".git/"))
                       for p in got)
        assert "src/pkg/.pytest_cache/v" not in got and not (wd / "src" / "__pycache__").exists()
        assert C.is_skipped_dir("data") and not C.is_skipped_dir("data", at_root=False)
        assert C.is_skipped_dir("__pycache__", at_root=False) and C.is_skipped_dir(".pytest_cache", at_root=False)
        assert C.SKIP_DIRS_ANY_DEPTH < C.SKIP_DIRS
    finally:
        C.cleanup_workdir(wd, keep=False)


def test_root_venv_and_host_data_are_skipped_only_at_root(monkeypatch, tmp_path):
    """scripts/setup_env.ps1이 저장소 루트에 만드는 .venv와 compose 기본 바인드 host-data/는 호스트 사본·전체 해시에서 빠진다.
    같은 이름의 하위 폴더는 그대로 들어간다(최상위 규칙)."""
    monkeypatch.delenv("NMFC_IN_CONTAINER", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    src = tmp_path / "src"
    _fake_repo(src)
    for rel in (".venv/Lib/site-packages/torch/big.bin", "host-data/cifar-100-python.tar.gz",
                "src/.venv/keep.py", "notes/host-data/keep.md"):
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_bytes(b"x")
    listed = set(C.list_files(src))
    assert not any(p.startswith((".venv/", "host-data/")) for p in listed)
    assert {"src/.venv/keep.py", "notes/host-data/keep.md"} <= listed
    wd = C.make_workdir(src)
    try:
        assert not (wd / ".venv").exists() and not (wd / "host-data").exists()
        assert (wd / "src" / ".venv" / "keep.py").exists() and (wd / "notes" / "host-data" / "keep.md").exists()
    finally:
        C.cleanup_workdir(wd, keep=False)


def test_make_workdir_in_container_returns_src_and_cleanup_never_deletes_it(monkeypatch, tmp_path):
    monkeypatch.setenv("NMFC_IN_CONTAINER", "1")
    src = tmp_path / "workspace"
    _fake_repo(src)
    assert C.make_workdir(src) == src
    C.cleanup_workdir(src, keep=False)
    assert (src / "README.md").exists()


def test_cleanup_workdir_keep_and_safety(monkeypatch, tmp_path):
    monkeypatch.delenv("NMFC_IN_CONTAINER", raising=False)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    src = tmp_path / "src"
    _fake_repo(src)
    C.cleanup_workdir(src, keep=False)                                  # 사본이 아닌 경로는 지우지 않는다
    assert (src / "README.md").exists()
    other = tmp_path / "tmp" / "not-a-workdir"
    other.mkdir()
    C.cleanup_workdir(other, keep=False)
    assert other.exists()

    wd = C.make_workdir(src, keep=True)
    C.cleanup_workdir(wd, keep=True)
    assert wd.exists()
    C.cleanup_workdir(wd, keep=False)
    assert not wd.exists()


# ---------------------------------------------------------------------------
# run_step
# ---------------------------------------------------------------------------
def test_run_step_expect_success(tmp_path):
    s = _script(tmp_path, "figs.py", """
        for i in range(14):
            print(f"saved: fig{i}.png")
        print("건너뜀 4건")
    """)
    r = C.run_step("figs", [str(s)], cwd=tmp_path, expect={"saved:": 14, "건너뜀 4건": 1})
    assert r.ok, r.notes
    assert r.exit_code == 0
    assert r.cmd[:2] == [sys.executable, "-B"] and r.cmd[2] == str(s)
    assert r.output_tail[-1] == "건너뜀 4건"
    assert r.output.count("saved:") == 14
    assert r.duration_s >= 0
    assert any("'saved:' >= 14회: 14회" in n for n in r.notes)


def test_run_step_expect_shortfall_fails(tmp_path):
    s = _script(tmp_path, "figs.py", """
        for i in range(13):
            print(f"saved: fig{i}.png")
    """)
    r = C.run_step("figs", [str(s)], cwd=tmp_path, expect={"saved:": 14})
    assert r.exit_code == 0
    assert not r.ok
    assert any("13회" in n and "부족" in n for n in r.notes)
    r2 = C.run_step("figs", [str(s)], cwd=tmp_path, expect={"OK: 없는 문자열": 1})
    assert not r2.ok


def test_run_step_counts_full_output_not_only_tail(tmp_path):
    s = _script(tmp_path, "long.py", """
        print("97 passed in 3.2s")
        for i in range(100):
            print("line", i)
    """)
    r = C.run_step("pytest", [str(s)], cwd=tmp_path, expect={"97 passed": 1})
    assert len(r.output_tail) == 40 and r.output_tail[-1] == "line 99"
    assert "97 passed" not in "\n".join(r.output_tail)
    assert r.ok


def test_run_step_merges_stderr(tmp_path):
    s = _script(tmp_path, "err.py", """
        import sys
        print("정합 검사 통과.", file=sys.stderr)
        print("stdout line")
    """)
    r = C.run_step("ledger", [str(s)], cwd=tmp_path, expect={"정합 검사 통과.": 1, "stdout line": 1})
    assert r.ok, r.notes


def test_run_step_exit_code_rules(tmp_path):
    s = _script(tmp_path, "exit1.py", """
        print("통과 16/16")
        raise SystemExit(1)
    """)
    r = C.run_step("a", [str(s)], cwd=tmp_path, expect={"통과 16/16": 1})
    assert r.exit_code == 1 and not r.ok                                # 기대 문자열이 있어도 exit가 틀리면 FAIL
    assert any("exit 1" in n for n in r.notes)
    r2 = C.run_step("a", [str(s)], cwd=tmp_path, allowed_exit=(0, 1), expect={"통과 16/16": 1})
    assert r2.ok
    r3 = C.run_step("a", [str(s)], cwd=tmp_path, allowed_exit=[0, 1], expect={"통과 16/16": 2})
    assert not r3.ok


def test_run_step_env_is_merged_and_cwd_used(tmp_path):
    s = _script(tmp_path, "env.py", """
        import os, sys
        print("VAR=" + os.environ.get("NMFC_T", "-"))
        print("UTF8=" + str(sys.flags.utf8_mode))
        print("PATH_SET=" + str(bool(os.environ.get("PATH"))))
        print("CWD=" + os.path.basename(os.getcwd()))
        print("PYC=" + str(sys.dont_write_bytecode))
    """)
    sub = tmp_path / "cwd_here"
    sub.mkdir()
    r = C.run_step("env", [str(s)], cwd=sub, env={"NMFC_T": "v1"},
                   expect={"VAR=v1": 1, "UTF8=1": 1, "PATH_SET=True": 1, "CWD=cwd_here": 1, "PYC=True": 1})
    assert r.ok, (r.notes, r.output_tail)


def test_run_step_python_false_and_missing_executable(tmp_path):
    r = C.run_step("raw", [sys.executable, "-c", "print('raw ok')"], cwd=tmp_path, python=False,
                   expect={"raw ok": 1})
    assert r.ok and r.cmd[0] == sys.executable and r.cmd[1] == "-c"
    r2 = C.run_step("missing", ["nmfc-no-such-binary-xyz", "--version"], cwd=tmp_path, python=False)
    assert not r2.ok and r2.exit_code == 127
    assert any("실행 실패" in n for n in r2.notes)


def test_run_step_timeout_kills_process_tree(tmp_path):
    s = _script(tmp_path, "hang.py", """
        import subprocess, sys, time
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], stdout=sys.stdout, stderr=sys.stderr)
        print("spawned", flush=True)
        time.sleep(60)
    """)
    r = C.run_step("hang", [str(s)], cwd=tmp_path, timeout=3, expect={"spawned": 1})
    assert not r.ok and r.exit_code == -1
    assert any("시간 초과" in n for n in r.notes)
    assert "spawned" in r.output
    assert r.duration_s < 9, r.duration_s                               # 손자가 파이프를 잡으면 ~13초가 된다


def test_run_step_echo_streams_output(tmp_path, capsys):
    s = _script(tmp_path, "echo.py", "print('실시간 줄')\n")
    r = C.run_step("echo", [str(s)], cwd=tmp_path, echo=True, expect={"실시간 줄": 1})
    assert r.ok
    out = capsys.readouterr().out
    assert "실시간 줄" in out and "[nmfc] >> echo" in out and "[nmfc] << echo" in out


# --- cp949 부모 회귀 -------------------------------------------------------
@pytest.fixture(scope="module")
def cp949_parent_result(tmp_path_factory):
    d = tmp_path_factory.mktemp("cp949")
    child_bytes = _script(d, "child_bytes.py", """
        import sys
        sys.stdout.buffer.write("통과 16/16\\n".encode("utf-8"))
        sys.stdout.buffer.flush()
    """)
    child_print = _script(d, "child_print.py", """
        print("통과 16/16")
        print("정합 검사 통과.")
    """)
    parent = _script(d, "parent.py", """
        import json, locale, subprocess, sys
        docker_dir, child_bytes, child_print, cwd, out = sys.argv[1:6]
        sys.path.insert(0, docker_dir)
        import nmfc_common as C
        res = {"enc": locale.getpreferredencoding(False), "utf8_mode": sys.flags.utf8_mode}
        r1 = C.run_step("bytes", [child_bytes], cwd=cwd, expect={"통과 16/16": 1})
        r2 = C.run_step("print", [child_print], cwd=cwd, expect={"통과 16/16": 1, "정합 검사 통과.": 1})
        res["r1"] = {"ok": r1.ok, "tail": r1.output_tail, "notes": r1.notes}
        res["r2"] = {"ok": r2.ok, "tail": r2.output_tail, "notes": r2.notes}
        try:   # 대조군: 인코딩을 지정하지 않은 text=True (부모 로캘 인코딩으로 디코딩)
            p = subprocess.run([sys.executable, "-B", child_bytes], capture_output=True, text=True)
            res["naive"] = {"got": "통과 16/16" in (p.stdout or ""), "error": None}
        except Exception as e:
            res["naive"] = {"got": False, "error": type(e).__name__}
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=True)
    """)
    out = d / "result.json"
    p = subprocess.run([sys.executable, "-B", str(parent), str(DOCKER_DIR), str(child_bytes), str(child_print),
                        str(d), str(out)], env=_plain_env(), capture_output=True, encoding="utf-8",
                       errors="replace", timeout=180)
    assert p.returncode == 0, p.stdout + p.stderr
    return json.loads(out.read_text(encoding="utf-8"))


def test_cp949_parent_reads_utf8_child_output(cp949_parent_result):
    res = cp949_parent_result
    assert res["utf8_mode"] == 0                                        # 부모는 UTF-8 모드가 아니다
    assert res["r1"]["ok"] and res["r1"]["tail"] == ["통과 16/16"], res["r1"]
    # 자식이 평범한 print를 써도 child_env(PYTHONUTF8=1) 덕분에 UTF-8로 나오고 깨지지 않는다
    assert res["r2"]["ok"] and res["r2"]["tail"] == ["통과 16/16", "정합 검사 통과."], res["r2"]


def test_cp949_parent_naive_decoding_breaks_control(cp949_parent_result):
    res = cp949_parent_result
    if codecs.lookup(res["enc"]).name == "utf-8":
        pytest.skip(f"부모 로캘 인코딩이 UTF-8({res['enc']})이라 대조군이 성립하지 않는다")
    assert not res["naive"]["got"], res                                 # 인코딩 미지정이면 깨진다(테스트가 실효적임)


# ---------------------------------------------------------------------------
# 보고서
# ---------------------------------------------------------------------------
def _step(name="s0", ok=True):
    return C.StepResult(name, [sys.executable, "-B", "-m", "pytest", "tests", "-q"], 0 if ok else 1, 1.234,
                        ["97 passed"], ok, ["기대 '97 passed' >= 1회: 1회"])


def test_report_write_schema_and_markdown(monkeypatch, tmp_path):
    monkeypatch.setenv("NMFC_MANIFEST_SHA256", "f" * 64)
    rep = C.Report("verify", tmp_path / "out")
    rep.set_env({"python": "3.14.6", "where": Path("a/b"), "cc": (12, 0)})
    rep.add_step(_step())
    rep.add_check("한글 폰트 단언", "PASS", "Malgun Gothic",
                  {"path": Path("c:/w/x.ttf"), "n": np.int64(3), "r": np.float32(0.5), "s": {"b", "a"},
                   "arr": np.arange(3), (1, 2): "tuple-key"})
    rep.add_check("ko 그림", "REPORT", "font_dependent|차이 0.1\n둘째 줄")
    rep.add_check("ckpt 평가", "N/A", "ckpt 없음")
    rep.add_check("밴드", "WARN", "rounding_boundary")
    rep.set_label("PASS_EXACT")
    assert rep.passed
    jp, mp = rep.write()
    assert jp == tmp_path / "out" / "verify_report.json" and mp == tmp_path / "out" / "verify_report.md"
    raw = jp.read_bytes()
    assert b"\r\n" not in raw
    d = json.loads(raw.decode("utf-8"))
    assert list(d) == ["command", "label", "pass", "manifest_sha256", "env", "started", "finished", "steps", "checks"]
    assert d["command"] == "verify" and d["label"] == "PASS_EXACT" and d["pass"] is True
    assert d["manifest_sha256"] == "f" * 64
    datetime.fromisoformat(d["started"])
    datetime.fromisoformat(d["finished"])
    assert set(d["steps"][0]) == {"name", "cmd", "exit_code", "duration_s", "output_tail", "ok", "notes"}
    assert set(d["checks"][0]) == {"name", "status", "detail", "data"}
    data = d["checks"][0]["data"]
    assert data["n"] == 3 and data["r"] == 0.5 and data["s"] == ["a", "b"] and data["arr"] == [0, 1, 2]
    assert data["path"] == "c:/w/x.ttf" and data["(1, 2)"] == "tuple-key"
    assert d["env"]["where"] == "a/b" and d["env"]["cc"] == [12, 0]

    md = mp.read_text(encoding="utf-8")
    assert md.splitlines()[0] == "nmfc.manifest.sha256: " + "f" * 64
    assert "한글 폰트 단언" in md and "PASS_EXACT" in md and "| 0 | s0 |" in md
    assert "font_dependent\\|차이 0.1<br>둘째 줄" in md                 # 표 셀 이스케이프
    assert b"\r\n" not in mp.read_bytes()


def test_report_pass_fail_logic(monkeypatch, tmp_path):
    monkeypatch.delenv("NMFC_MANIFEST_SHA256", raising=False)
    monkeypatch.setattr(C, "REPO_ROOT", tmp_path)                      # lock 없음 → manifest_sha256 None
    rep = C.Report("regen", tmp_path)
    rep.add_step(_step())
    rep.add_check("x", "PASS")
    assert rep.passed
    rep.add_check("purity", "REPORT", "보고만")
    assert rep.passed
    rep.add_check("mAP", "FAIL", "abs 2e-5 > 1e-5")
    assert not rep.passed
    jp, mp = rep.write()
    assert json.loads(jp.read_text(encoding="utf-8"))["pass"] is False
    assert json.loads(jp.read_text(encoding="utf-8"))["manifest_sha256"] is None
    assert mp.read_text(encoding="utf-8").splitlines()[0] == "nmfc.manifest.sha256: none"

    rep2 = C.Report("smoke", tmp_path)
    rep2.add_step(_step("bad", ok=False))
    assert not rep2.passed
    md = rep2.to_markdown()
    assert "실패한 단계의 출력 끝부분" in md and "97 passed" in md

    rep3 = C.Report("regen", tmp_path)
    rep3.set_label("FAIL")                                              # 라벨 FAIL이면 pass도 false
    assert not rep3.passed
    with pytest.raises(ValueError):
        rep3.add_check("x", "OK")


def test_report_command_name_is_sanitized(tmp_path):
    jp, mp = C.Report("fetch data", tmp_path).write()
    assert jp.name == "fetch_data_report.json" and mp.name == "fetch_data_report.md"
    assert json.loads(jp.read_text(encoding="utf-8"))["command"] == "fetch data"


# ---------------------------------------------------------------------------
# import 규칙과 장치 정보 (다른 테스트에 영향을 주지 않도록 별도 프로세스에서)
# ---------------------------------------------------------------------------
def test_top_level_imports_stay_light():
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import nmfc_common, nmfc_manifest; "
            "heavy = ['torch', 'torchvision', 'matplotlib', 'PIL', 'scipy']; "
            "print('HEAVY=' + ','.join(m for m in heavy if m in sys.modules))")
    p = subprocess.run([sys.executable, "-B", "-c", code, str(DOCKER_DIR)], capture_output=True, encoding="utf-8",
                       errors="replace", env=C.child_env(), timeout=120)
    assert p.returncode == 0, p.stderr
    assert "HEAVY=\n" in p.stdout or p.stdout.strip() == "HEAVY=", p.stdout


def test_device_info_shape():
    code = ("import sys, json; sys.path.insert(0, sys.argv[1]); import nmfc_common as C; "
            "print('JSON=' + json.dumps(C.device_info()))")
    p = subprocess.run([sys.executable, "-B", "-c", code, str(DOCKER_DIR)], capture_output=True, encoding="utf-8",
                       errors="replace", env=C.child_env(), timeout=300)
    assert p.returncode == 0, p.stderr
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("JSON=")][-1]
    info = json.loads(line[len("JSON="):])
    assert {"torch", "cuda", "cudnn", "gpu_name", "capability", "gpu_mem", "device"} <= set(info)
    assert info["device"] in ("cpu", "cuda")
    if info["device"] == "cuda":
        assert isinstance(info["gpu_name"], str) and isinstance(info["gpu_mem"], int)
        assert isinstance(info["capability"], list) and len(info["capability"]) == 2
