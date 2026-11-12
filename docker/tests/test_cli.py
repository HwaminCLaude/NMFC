"""nmfc_cli 통합 진입점.

무거운 실제 실행(verify·regen·smoke·train 학습)은 여기서 돌리지 않는다. 여기서는 빠르게 돌 수 있는 것만 본다.
- 디스패치: --help, 하위 명령 도움말, manifest 그대로 넘기기(torch·matplotlib·numpy를 import하지 않음)
- 판정 도우미: 예상 밖 변경 비교, lock 결과 필터, JSON 표지 파싱, 종료코드↔보고서 pass 일관성
- fetch: --from 복사·sha256·tar filter='data'(상위 경로 탈출 거부)·손상 파일 FAIL·.part 정리
- 작업 트리 가드(원본 저장소에 쓰는 --workdir 거부), compare 전제조건(상태 없음 → 2), 상태 이력 합치기
- train --dry-run이 아무것도 쓰지 않음
모든 판정기는 '일부러 망가뜨린 입력에서 FAIL'을 포함한다.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

DOCKER = Path(__file__).resolve().parents[1]
REPO = DOCKER.parent
if str(DOCKER) not in sys.path:
    sys.path.insert(0, str(DOCKER))

import nmfc_cli as CLI  # noqa: E402
import nmfc_common as common  # noqa: E402
import nmfc_pipeline as P  # noqa: E402

COMMANDS = ["env", "verify", "regen", "fetch", "smoke", "train", "reproduce", "compare", "manifest"]


@pytest.fixture(autouse=True)
def _host(monkeypatch):
    monkeypatch.delenv("NMFC_IN_CONTAINER", raising=False)
    monkeypatch.delenv("NMFC_BASELINE", raising=False)
    monkeypatch.delenv("NMFC_MANIFEST_SHA256", raising=False)


def _report(out: Path, name: str) -> dict:
    return json.loads((out / f"{name}_report.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 디스패치

def test_help_lists_all_commands(capsys):
    assert CLI.main(["--help"]) == 0
    out = capsys.readouterr().out
    for c in COMMANDS:
        assert c in out
    assert "_smoke-child" not in out and "_device-info" not in out        # 숨은 명령은 도움말에 없다


@pytest.mark.parametrize("cmd", COMMANDS)
def test_each_command_help_exit0(cmd, capsys):
    assert CLI.main([cmd, "--help"]) == 0


def test_unknown_command_and_bad_choice_exit2(capsys):
    assert CLI.main(["bogus"]) == 2
    capsys.readouterr()
    assert CLI.main(["paper", "--help"]) == 2                            # 제거된 하위 명령: 도움말도 없다
    assert "invalid choice: 'paper'" in capsys.readouterr().err
    assert CLI.main(["regen", "--device", "tpu"]) == 2
    assert CLI.main([]) == 2


def test_help_and_manifest_do_not_import_heavy_modules(tmp_path):
    """--help와 manifest(빌드 중 finalize가 부르는 경로)는 torch·matplotlib·numpy 없이 돌아야 한다."""
    ctx = tmp_path / "ctx"
    (ctx / "docker").mkdir(parents=True)
    (ctx / "a.txt").write_bytes(b"x\r\n")
    probe = ("import sys; sys.path.insert(0, sys.argv[1]); import nmfc_cli; "
             "rc = nmfc_cli.main(sys.argv[2:]); "
             "print('HEAVY', sorted(m for m in ('torch', 'matplotlib', 'numpy') if m in sys.modules)); sys.exit(rc)")
    env = common.child_env()
    for args, want in ((["--help"], 0), (["manifest", "--write-lock", str(ctx)], 0),
                       (["manifest", "--verify", str(ctx / "docker/context.lock"), "--root", str(ctx), "--strict"], 0)):
        r = subprocess.run([sys.executable, "-B", "-c", probe, str(DOCKER), *args], capture_output=True,
                           encoding="utf-8", errors="replace", env=env, timeout=120)
        assert r.returncode == want, r.stdout + r.stderr
        assert "HEAVY []" in r.stdout, r.stdout


def test_manifest_passthrough_and_tamper_fails(tmp_path, capsys):
    ctx = tmp_path / "ctx"
    (ctx / "results").mkdir(parents=True)
    (ctx / "results" / "m.json").write_text('{"a": 1}\n', encoding="utf-8")
    assert CLI.main(["manifest", "--write-lock", str(ctx)]) == 0
    lock = ctx / "docker" / "context.lock"
    assert lock.is_file()
    assert CLI.main(["manifest", "--verify", str(lock), "--root", str(ctx), "--strict"]) == 0
    (ctx / "results" / "new.json").write_text("{}", encoding="utf-8")            # 망가뜨림: 새 파일
    assert CLI.main(["manifest", "--verify", str(lock), "--root", str(ctx), "--strict"]) == 1
    assert CLI.main(["manifest", "--verify", str(lock), "--root", str(ctx)]) == 0  # 비엄격은 results/ 새 파일 무시
    dest = tmp_path / "bl"
    assert CLI.main(["manifest", "--baseline", str(dest), "--root", str(ctx)]) == 0  # 공통 --baseline과 충돌 없음
    assert (dest / "results" / "m.json").is_file()


# ---------------------------------------------------------------- 판정 도우미

def test_unexpected_changes_detects_and_allows_writes():
    pre = {"a.txt": "1", "results/x.json": "2", "fig.png": "3", "gone.txt": "4"}
    post = {"a.txt": "1", "results/x.json": "9", "fig.png": "8", "results/new.json": "5", "newfig.png": "6"}
    bad = CLI.unexpected_changes(pre, post, {"fig.png", "newfig.png"})
    assert [(f["path"], f["kind"]) for f in bad] == [("gone.txt", "missing"), ("results/new.json", "extra"),
                                                    ("results/x.json", "changed")]
    assert CLI.unexpected_changes(pre, dict(pre), set()) == []
    # writes에 있어도 사라진 파일은 잡는다
    assert CLI.unexpected_changes({"fig.png": "1"}, {}, {"fig.png"})[0]["kind"] == "missing"


def test_filter_lock_findings_keeps_missing_and_foreign_changes():
    f = [{"path": "w.png", "kind": "changed"}, {"path": "w.png", "kind": "missing"},
         {"path": "other.py", "kind": "changed"}, {"path": "results/n.json", "kind": "extra"}]
    left = CLI.filter_lock_findings(f, {"w.png"})
    assert [(x["path"], x["kind"]) for x in left] == [("w.png", "missing"), ("other.py", "changed"),
                                                      ("results/n.json", "extra")]


def test_unexpected_changes_on_real_tree(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "a.md").write_bytes(b"a\r\nb\r\n")
    (tmp_path / "runs").mkdir()                                       # SKIP_DIRS는 보지 않는다
    pre = CLI._hash_tree(tmp_path)
    (tmp_path / "notes" / "a.md").write_bytes(b"a\nb\n")              # CRLF→LF만 바뀜: 같은 내용
    (tmp_path / "runs" / "status.json").write_text("{}", encoding="utf-8")
    assert CLI.unexpected_changes(pre, CLI._hash_tree(tmp_path), set()) == []
    (tmp_path / "notes" / "a.md").write_bytes(b"a\nB\n")              # 망가뜨림
    assert CLI.unexpected_changes(pre, CLI._hash_tree(tmp_path), set())[0]["kind"] == "changed"


def test_hash_tree_sees_new_files_under_nested_skip_names(tmp_path):
    """verify의 '예상 밖 변경'(lock 없는 호스트): 하위 폴더의 data/·runs/·out/ 아래 새 파일도 잡는다."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x", encoding="utf-8")
    pre = CLI._hash_tree(tmp_path)
    for rel in ("results/data/evil.json", "src/runs/evil.py", "logs/x/out/evil.png"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"{}")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "ok.bin").write_bytes(b"x")                     # 최상위 data/는 보지 않는다
    bad = CLI.unexpected_changes(pre, CLI._hash_tree(tmp_path), set())
    assert [(f["path"], f["kind"]) for f in bad] == [("logs/x/out/evil.png", "extra"),
                                                    ("results/data/evil.json", "extra"), ("src/runs/evil.py", "extra")]


def test_parse_json_mark_and_device_env():
    out = "noise\nNMFC_JSON={\"ok\": true, \"x\": 1}\ntrailing\n"
    assert CLI._parse_json_mark(out) == {"ok": True, "x": 1}
    assert CLI._parse_json_mark("NMFC_JSON={broken") is None
    assert CLI._parse_json_mark("") is None
    assert CLI._device_env("cpu") == {"CUDA_VISIBLE_DEVICES": "-1"}
    assert CLI._device_env("cuda") == {} and CLI._device_env("auto") == {}


def test_finish_keeps_pass_consistent_with_exit(tmp_path):
    rep = common.Report("compare", tmp_path)
    rep.add_check("x", "N/A", "미평가")
    assert CLI.finish(rep, 2) == 2
    d = _report(tmp_path, "compare")
    assert d["pass"] is False and d["label"] == "INCONCLUSIVE"
    rep2 = common.Report("verify", tmp_path / "b")
    rep2.add_check("y", "FAIL", "망가짐")
    assert CLI.finish(rep2, 0) == 1                                     # FAIL 검사가 있으면 0을 1로 바꾼다
    assert _report(tmp_path / "b", "verify")["label"] == "FAIL"


def test_regen_order():
    ids = list(CLI.REGEN_ORDER)
    assert all(i in P.STEP_BY_ID for i in ids)
    assert ids.index("S3.near_ood") < ids.index("S3.seed_robustness")    # seed_robustness가 near_ood.json을 읽는다
    assert ids[:3] == ["S3.retrieval", "S3.fewshot_verify", "S3.stats_ci"]
    assert ids[-1:] == ["S3.noise_grid"] and not [i for i in ids if i.startswith("S5")]
    ckpt = {i for i in ids if P.STEP_BY_ID[i]["needs"] or P.STEP_BY_ID[i]["needs_any"]}
    assert ckpt == {"S3.near_ood", "S1.bank_softmax", "S1.bank_shrb", "S3.calibration", "S3.openset_detection"}


def test_font_assert_fails_for_unknown_family():
    pytest.importorskip("matplotlib")
    st, detail, data = CLI.font_assert({"families": ["NoSuchFontFamily-nmfc-xyz"], "char": "한"})
    assert st == "FAIL" and data["found"] == []


# ---------------------------------------------------------------- fetch

def _assets(tmp_path: Path, items: dict) -> Path:
    p = tmp_path / "assets.json"
    p.write_text(json.dumps(items), encoding="utf-8")
    return p


def _tar_gz(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in members.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def test_fetch_weights_from_local_ok_and_corrupt(tmp_path):
    payload = b"weights" * 1000
    src = tmp_path / "src"
    src.mkdir()
    (src / "w.pth").write_bytes(payload)
    assets = _assets(tmp_path, {"weights": [{"name": "w.pth", "dest": "$TORCH_HOME/hub/checkpoints/w.pth",
                                             "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                                             "urls": []}]})
    dest, out = tmp_path / "dest", tmp_path / "out"
    rc = CLI.main(["fetch", "weights", "--from", str(src), "--dest", str(dest), "--assets", str(assets),
                   "--out", str(out)])
    assert rc == 0 and (dest / "w.pth").read_bytes() == payload
    # 망가뜨림: 원본이 잘렸다 → FAIL, 대상·.part 모두 남지 않는다
    (src / "w.pth").write_bytes(payload[:-1])
    dest2 = tmp_path / "dest2"
    rc = CLI.main(["fetch", "weights", "--from", str(src), "--dest", str(dest2), "--assets", str(assets),
                   "--out", str(out)])
    assert rc == 1
    assert not (dest2 / "w.pth").exists() and not (dest2 / "w.pth.part").exists()
    d = _report(out, "fetch_weights")
    assert any(c["status"] == "FAIL" and "불일치" in c["detail"] for c in d["checks"])


def test_fetch_data_tar_extracts_with_data_filter(tmp_path):
    blob = _tar_gz({"pkg/a.bin": b"A" * 10, "pkg/sub/b.txt": b"hello\n"})
    src = tmp_path / "src"
    src.mkdir()
    (src / "pkg.tar.gz").write_bytes(blob)
    assets = _assets(tmp_path, {"data": [{"name": "pkg.tar.gz", "dest": "data/raw/pkg.tar.gz", "size": len(blob),
                                          "sha256": hashlib.sha256(blob).hexdigest(), "md5": None, "urls": [],
                                          "extract": "tar.gz"}]})
    dest, out = tmp_path / "raw", tmp_path / "out"
    rc = CLI.main(["fetch", "data", "--from", str(src), "--dest", str(dest), "--assets", str(assets),
                   "--out", str(out), "--skip-integrity"])
    assert rc == 0
    assert (dest / "pkg" / "sub" / "b.txt").read_bytes() == b"hello\n"
    assert (src / "pkg.tar.gz").read_bytes() == blob                  # --from은 읽기만


def test_extract_tar_rejects_path_traversal(tmp_path):
    bad = tmp_path / "bad.tar.gz"
    bad.write_bytes(_tar_gz({"../evil.txt": b"x"}))
    dest = tmp_path / "d"
    dest.mkdir()
    with pytest.raises(tarfile.TarError):
        CLI.extract_tar(bad, dest)
    assert not (tmp_path / "evil.txt").exists()


def test_fetch_missing_from_dir_is_precondition(tmp_path):
    assert CLI.main(["fetch", "data", "--from", str(tmp_path / "nope"), "--dest", str(tmp_path / "d"),
                     "--out", str(tmp_path / "o")]) == 2


def _step(code: int) -> common.StepResult:
    return common.StepResult(name="tv", cmd=["x"], exit_code=code, duration_s=0.0, output_tail=["tail"], ok=code == 0)


def _good_integrity() -> dict:
    files = [{"path": f"f{i}", "md5": "0" * 32, "ok": True} for i in range(CLI.TV_INTEGRITY_FILES)]
    dsets = [{"name": f"d{i}", "ok": True, "len": 1} for i in range(CLI.TV_INTEGRITY_DATASETS)]
    return {"files": files, "datasets": dsets, "ok": True}


@pytest.mark.parametrize("code,res,why", [
    (1, {"files": [], "datasets": [], "ok": False, "error": "RuntimeError: operator torchvision::nms does not exist"},
     "검사 중 예외"),                                                    # import 실패: 항목 0건
    (0, {"files": [], "datasets": [], "ok": True}, "md5 검사 0건"),     # 자식은 ok라는데 검사 0건
    (0, {**_good_integrity(), "ok": False}, "ok=False"),
    (3, _good_integrity(), "자식 종료코드 3"),
    (0, {**_good_integrity(), "files": _good_integrity()["files"][:-1]}, "md5 검사 10건"),
    (0, None, "검사 실행 실패"),
])
def test_integrity_checks_fail_on_child_failure_signals(tmp_path, code, res, why):
    """fetch data: torchvision 무결성 자식이 실패를 알리면 항목 검사가 0건이어도 FAIL이어야 한다."""
    rep = common.Report("fetch data", tmp_path)
    CLI.integrity_checks(rep, _step(code), res)
    agg = [c for c in rep.checks if c["name"] == "torchvision 무결성"]
    assert len(agg) == 1 and agg[0]["status"] == "FAIL" and why in agg[0]["detail"], agg
    assert rep.passed is False


def test_integrity_checks_pass_only_when_complete(tmp_path):
    rep = common.Report("fetch data", tmp_path)
    CLI.integrity_checks(rep, _step(0), _good_integrity())
    assert rep.passed is True
    assert [c["status"] for c in rep.checks if c["name"] == "torchvision 무결성"] == ["PASS"]
    assert sum(c["name"].startswith("md5 ") for c in rep.checks) == 11


def _extracted_only_from(tmp_path: Path) -> tuple[Path, Path]:
    """--from에 압축본 없이 추출본(쓰레기 내용)과 sha256이 맞는 mat만 둔다. 실제 assets.json의 이름을 쓴다."""
    src = tmp_path / "from"
    for folder, fn in (("cifar-100-python", "train"), ("cifar-10-batches-py", "data_batch_1")):
        (src / folder).mkdir(parents=True)
        (src / folder / fn).write_bytes(b"garbage")
    mat = b"fake svhn" * 50
    (src / "test_32x32.mat").write_bytes(mat)
    real = json.loads((DOCKER / "assets.json").read_text(encoding="utf-8"))
    for it in real["data"]:
        if it["name"] == "test_32x32.mat":
            it["sha256"], it["size"] = hashlib.sha256(mat).hexdigest(), len(mat)
    return src, _assets(tmp_path, real)


def test_fetch_extracted_only_with_broken_torchvision_fails(tmp_path, monkeypatch):
    """추출본만 있는 --from + torchvision import 고장(ABI 불일치 흉내): 예전에는 검사 0건으로 PASS였다."""
    src, assets = _extracted_only_from(tmp_path)
    shim = tmp_path / "shim" / "torchvision"
    shim.mkdir(parents=True)
    (shim / "__init__.py").write_text("raise RuntimeError('operator torchvision::nms does not exist')\n",
                                      encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(shim.parent))                  # 자식 프로세스만 이 env를 받는다
    out = tmp_path / "out"
    rc = CLI.main(["fetch", "data", "--from", str(src), "--dest", str(tmp_path / "raw"), "--assets", str(assets),
                   "--out", str(out)])
    assert rc == 1
    d = _report(out, "fetch_data")
    agg = [c for c in d["checks"] if c["name"] == "torchvision 무결성"]
    assert agg and agg[0]["status"] == "FAIL" and "operator torchvision::nms" in agg[0]["detail"]
    assert [c["status"] for c in d["checks"] if c["name"].endswith(".tar.gz")] == ["REPORT", "REPORT"]
    assert any(s["name"].startswith("torchvision 무결성") and s["exit_code"] == 1 for s in d["steps"])
    # 추출본을 복사했는데 무결성 검사를 건너뛰면 검증된 것이 없다 → FAIL
    rc = CLI.main(["fetch", "data", "--from", str(src), "--dest", str(tmp_path / "raw2"), "--assets", str(assets),
                   "--out", str(tmp_path / "out2"), "--skip-integrity"])
    assert rc == 1


# ---------------------------------------------------------------- 작업 트리·compare·train

def test_writing_commands_refuse_repo_root_as_workdir(tmp_path):
    for cmd in (["verify"], ["regen", "--device", "cpu"]):
        out = tmp_path / cmd[0]
        rc = CLI.main([*cmd, "--workdir", str(REPO), "--out", str(out)])
        assert rc == 2, cmd
        d = _report(out, cmd[0])
        assert d["pass"] is False and any("원본 저장소" in c["detail"] for c in d["checks"])


def test_compare_without_status_is_precondition(tmp_path):
    wd = tmp_path / "wd"
    wd.mkdir()
    out = tmp_path / "out"
    assert CLI.main(["compare", "--workdir", str(wd), "--out", str(out)]) == 2
    d = _report(out, "compare")
    assert d["pass"] is False and d["label"] == "INCONCLUSIVE"


def test_status_history_archive_and_merge(tmp_path):
    wd = tmp_path
    (wd / "runs").mkdir()
    s1 = {"started": "2026-11-10T10:00:00+09:00", "started_ns": 100, "trained": ["syn_a", "syn_b"],
          "items": [{"id": "T1.1", "trained": ["syn_a", "syn_b"], "started_ns": 110}]}
    (wd / "runs" / "pipeline_status.json").write_text(json.dumps(s1), encoding="utf-8")
    arch = CLI.archive_status(wd)
    assert arch is not None and arch.name == "pipeline_status.100.json"
    assert CLI.archive_status(wd) == arch                               # 같은 실행은 한 번만 보관
    s2 = {"started": "2026-11-10T11:00:00+09:00", "started_ns": 200, "trained": ["c100v_shrb"],
          "items": [{"id": "T5:c100v_shrb", "trained": ["c100v_shrb"], "started_ns": 210}]}
    (wd / "runs" / "pipeline_status.json").write_text(json.dumps(s2), encoding="utf-8")
    merged, src = CLI.merged_status(wd)
    assert merged["trained"] == ["syn_a", "syn_b", "c100v_shrb"]
    assert [i["id"] for i in merged["items"]] == ["T1.1", "T5:c100v_shrb"]
    assert merged["started_ns"] == 100 and len(src) == 2
    latest, src1 = CLI.merged_status(wd, latest_only=True)
    assert latest["trained"] == ["c100v_shrb"] and len(src1) == 1
    # 현재 상태와 같은 실행의 보관본은 두 번 세지 않는다
    CLI.archive_status(wd)
    merged2, _ = CLI.merged_status(wd)
    assert merged2["trained"] == ["syn_a", "syn_b", "c100v_shrb"] and len(merged2["items"]) == 2


def _trained_workdir(tmp_path: Path, exp: str = "syn_moons_softmax") -> tuple[Path, dict]:
    """합성 1런을 '학습한' 작업 트리: 바뀐 metrics.json, 학습 전 기준본(runs/nmfc_baseline), 성공한 상태 JSON."""
    import time
    wd = tmp_path / "wd"
    ref = REPO / "results" / exp / "metrics.json"
    m = json.loads(ref.read_text(encoding="utf-8"))
    m["silhouette"] = m["silhouette"] - 0.001
    (wd / "results" / exp).mkdir(parents=True)
    (wd / "results" / exp / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    (wd / CLI.HOST_TRAIN_BASELINE_REL / "results" / exp).mkdir(parents=True)
    shutil.copyfile(ref, wd / CLI.HOST_TRAIN_BASELINE_REL / "results" / exp / "metrics.json")
    t = time.time_ns() - 60 * 10**9
    ok = {"started": "2026-11-10T10:00:00+09:00", "started_ns": t, "finished": "x", "pass": True, "exit_code": 0,
          "args": {"exp": exp, "force": True}, "trained": [exp],
          "items": [{"id": f"T1:{exp}", "kind": "train", "required": True, "exit": 0, "ok": True, "expected": [exp],
                     "cached": [], "trained": [exp], "started_ns": t, "artifacts": {exp: P.run_artifacts(wd, exp, None)}}]}
    (wd / "runs" / "pipeline_status.json").write_text(json.dumps(ok), encoding="utf-8")
    return wd, ok


def test_compare_after_failed_latest_train_is_not_pass(tmp_path):
    """마지막 train이 실패(학습 0런)했는데 이력 합치기로 compare가 PASS하던 결함."""
    wd, ok = _trained_workdir(tmp_path)
    out = tmp_path / "o1"
    assert CLI.main(["compare", "--workdir", str(wd), "--out", str(out)]) == 0          # 대조: 성공 1회 → PASS
    assert CLI.archive_status(wd) is not None
    bad = {"started": "2026-11-10T11:00:00+09:00", "started_ns": ok["started_ns"] + 10**9, "finished": "y",
           "pass": False, "exit_code": 1, "args": {"exp": "c100v_shrb", "force": True}, "trained": [],
           "items": [{"id": "T5:c100v_shrb", "kind": "train", "required": True, "exit": 1, "ok": False,
                      "expected": ["c100v_shrb"], "cached": [], "trained": [], "started_ns": ok["started_ns"] + 10**9},
                     {"id": "S3.retrieval", "kind": "step", "required": True, "exit": None, "ok": None,
                      "skipped_reason": "학습 실패 — 후속 단계를 실행하지 않음"}]}
    (wd / "runs" / "pipeline_status.json").write_text(json.dumps(bad), encoding="utf-8")
    out = tmp_path / "o2"
    assert CLI.main(["compare", "--workdir", str(wd), "--out", str(out)]) == 2
    d = _report(out, "compare")
    assert d["pass"] is False and d["label"] == "INCONCLUSIVE"
    runs = [c for c in d["checks"] if c["name"].startswith("실행 ")]
    assert [c["status"] for c in runs] == ["REPORT", "FAIL"] and "c100v_shrb" in runs[1]["detail"]
    pre = next(c for c in d["checks"] if c["name"].startswith("전제조건"))
    assert pre["status"] == "FAIL" and "마지막 train/reproduce 실행이 성공하지 않음" in pre["detail"]
    assert CLI.main(["compare", "--workdir", str(wd), "--latest-only", "--out", str(tmp_path / "o3")]) == 2
    merged, sources = CLI.merged_status(wd)
    assert [r.get("pass") for r in merged["runs"]] == [True, False] and len(sources) == 2
    assert [r["_source"] for r in merged["runs"]] == sources


def test_compare_detects_metrics_restored_after_training(tmp_path):
    """학습 뒤 metrics.json을 학습 전 원본으로 되돌리면(새 mtime) Δ=0 PASS가 나던 결함."""
    wd, _ = _trained_workdir(tmp_path)
    shutil.copyfile(wd / CLI.HOST_TRAIN_BASELINE_REL / "results" / "syn_moons_softmax" / "metrics.json",
                    wd / "results" / "syn_moons_softmax" / "metrics.json")
    out = tmp_path / "o"
    assert CLI.main(["compare", "--workdir", str(wd), "--out", str(out)]) == 2
    pre = next(c for c in _report(out, "compare")["checks"] if c["name"].startswith("전제조건"))
    assert pre["status"] == "FAIL" and "학습 직후 기록" in pre["detail"]


def test_train_dry_run_writes_nothing(tmp_path, capsys):
    runs_existed = (REPO / "runs").exists()
    out = tmp_path / "out"
    assert CLI.main(["train", "--stage", "T1", "--dry-run", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "[dry-run]" in text and "syn_moons_linear" in text
    assert (REPO / "runs").exists() == runs_existed
    assert _report(out, "train")["pass"] is True
    assert CLI.main(["train", "--exp", "no_such_exp_xyz", "--dry-run", "--out", str(out)]) == 2


def test_pipeline_child_env_forces_utf8(monkeypatch):
    monkeypatch.setenv("PYTHONIOENCODING", "cp949")
    env = P.child_env({"PYTHONUTF8": "0", "X_NMFC": "1"})
    assert env["PYTHONIOENCODING"] == "utf-8" and env["PYTHONUTF8"] == "1" and env["X_NMFC"] == "1"


def test_smoke_dataset_shapes():
    torch = pytest.importorskip("torch")
    ds = CLI.SmokeImageDataset(5, size=8, classes=3)
    x, y = ds[4]
    assert len(ds) == 5 and tuple(x.shape) == (3, 8, 8) and x.dtype == torch.float32 and y == 1
    with pytest.raises(IndexError):
        ds[5]
