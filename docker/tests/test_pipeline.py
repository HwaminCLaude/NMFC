"""nmfc_pipeline 동작.

1) 원본 저장소에 대한 dry-run 계획: 명령 순서(near_ood가 seed_robustness 앞),
   --exp c100v_shrb → cifar100_variants.yaml --only c100v_shrb, --stage T1 → 9런, --then aggregate 구성.
   dry-run은 아무것도 쓰지 않는다(원본 저장소에 runs/가 생기지 않는지 확인).
2) 허용 exit 정책 단위 테스트와 STEPS ↔ ps1 대조.
3) 스텁 저장소(tmp_path)에서 실제 실행기를 돌린다: mtime 학습 판정, 사후조건, 실패 정책, 상태 JSON, --out,
   S2 래퍼. 진짜 학습·평가는 돌리지 않는다(src/train.py·scripts/*.py를 가짜로 바꾼 사본).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

DOCKER = Path(__file__).resolve().parents[1]
REPO = DOCKER.parent
if str(DOCKER) not in sys.path:
    sys.path.insert(0, str(DOCKER))

import nmfc_pipeline as P  # noqa: E402

DRY_RE = re.compile(r"^\[dry-run\]\s+\d+\s+(\S+)\s+(run|skip)\s+python -B (.*)$", re.M)
T1_CONFIG_ORDER = ["syn_moons_linear", "syn_circles_linear", "syn_swiss_linear", "syn_moons_softmax",
                   "syn_moons_nmfc", "syn_circles_softmax", "syn_circles_nmfc", "syn_swiss_softmax",
                   "syn_swiss_nmfc"]
AGG = ["S3.retrieval", "S3.fewshot_verify", "S3.stats_ci", "S3.near_ood", "S3.seed_robustness"]


@pytest.fixture
def repo_untouched():
    runs = REPO / "runs"
    existed = runs.exists()
    yield
    assert runs.exists() == existed, "dry-run이 원본 저장소에 runs/를 만들었다"


def dry(capsys, **kw):
    rc = P.run_pipeline(REPO, dry_run=True, **kw)
    out = capsys.readouterr().out
    return rc, [(m[1], m[2], m[3]) for m in DRY_RE.finditer(out)], out


# ---------------------------------------------------------------- 1) dry-run 계획 (원본 저장소, 읽기만)

def test_full_plan_order(capsys, repo_untouched):
    rc, rows, out = dry(capsys)
    assert rc == 0
    ids = [r[0] for r in rows]
    train_ids = (["T1.1", "T2.1", "T3.1", "T4.1"] + [f"T5.{i}" for i in range(1, 8)]
                 + [f"T6.{i}" for i in range(1, 9)] + ["T7.1", "T7.2"] + [f"T8.{i}" for i in range(1, 7)])
    assert ids == train_ids + [s["id"] for s in P.STEPS]
    assert ids.index("S3.near_ood") < ids.index("S3.seed_robustness")          # ps1:65-66과 반대 순서
    assert ids[-1] == "S4.theory" and not [i for i in ids if i.startswith("S5")]   # 학습 뒤 S1~S4만
    assert "S4.conditions" not in ids
    assert "큐 대조(ps1): OK" in out and "합계: 항목 40개, 학습 런 127개" in out
    states = dict((r[0], r[1]) for r in rows)
    assert states["S2.ext"] == "skip"                                           # --with-ext 아님
    assert "--with-ext 미지정" in out


def test_exp_resolves_to_config_with_only(capsys, repo_untouched):
    assert P.find_exp("c100v_shrb", REPO) == ("T5", "configs/cifar100_variants.yaml")
    rc, rows, out = dry(capsys, exp="c100v_shrb", force=True)
    assert rc == 0
    assert rows == [("T5:c100v_shrb", "run",
                     "-m src.train --config configs/cifar100_variants.yaml --only c100v_shrb --force")]
    assert "학습 런 1개" in out
    plan = P.build_plan(REPO, exp="c100v_shrb")
    assert plan[0]["cmd"] == ["-m", "src.train", "--config", "configs/cifar100_variants.yaml",
                              "--only", "c100v_shrb"] and plan[0]["expected"] == ["c100v_shrb"]
    assert P.find_exp("ood10v_shrb_s3", REPO) == ("T6", "configs/ood_cifar10_seeds_s3.yaml")
    assert P.find_exp("c100v_softmax_open_na15_s2", REPO) == ("T8", "configs/cifar100_openset_noise_grid_s2.yaml")


def test_stage_T1_is_9_runs(capsys, repo_untouched):
    plan = P.build_plan(REPO, stages=["T1"], force=True)
    assert len(plan) == 1 and plan[0]["id"] == "T1.1"
    assert plan[0]["expected"] == T1_CONFIG_ORDER and len(plan[0]["expected"]) == 9
    assert plan[0]["cmd"][:4] == ["-m", "src.train", "--config", "configs/synthetic.yaml"]
    assert plan[0]["cmd"][-1] == "--force"
    rc, rows, out = dry(capsys, stages=["T1"], force=True)
    assert rc == 0 and [r[0] for r in rows] == ["T1.1"]
    assert "합계: 항목 1개, 학습 런 9개" in out
    assert len(P.build_plan(REPO, stages=["t6", "T8"])) == 14                  # 소문자·여러 단계


def test_then_aggregate_composition(capsys, repo_untouched):
    assert list(P.AGGREGATE_IDS) == AGG
    plan = P.build_plan(REPO, exp="c100v_shrb", force=True, then_aggregate=True)
    assert [p["id"] for p in plan] == ["T5:c100v_shrb"] + AGG
    rc, rows, _ = dry(capsys, exp="c100v_shrb", force=True, then_aggregate=True)
    assert rc == 0 and [r[0] for r in rows] == ["T5:c100v_shrb"] + AGG
    # near_ood의 run/skip은 ckpt 유무로 갈린다(환경 의존) — 어느 쪽이든 자리는 seed_robustness 앞 그대로
    has_ckpt = any((REPO / f"checkpoints/{e}/ckpt.pt").exists() for e in P.NEAR_OOD_EXPS)
    assert dict((r[0], r[1]) for r in rows)["S3.near_ood"] == ("run" if has_ckpt else "skip")
    assert [p["id"] for p in P.build_plan(REPO, stages=["T1"], then_aggregate=True)] == ["T1.1"] + AGG
    assert [p["id"] for p in P.build_plan(REPO, stages=["T1"])] == ["T1.1"]    # train 모드는 학습만


def test_from_to_slicing():
    s3 = [s["id"] for s in P.STEPS if s["group"] == "S3"]
    assert [p["id"] for p in P.build_plan(REPO, from_step="S3", to_step="S3")] == s3
    assert [p["id"] for p in P.build_plan(REPO, from_step="S3.near_ood", to_step="S4.regime")] == [
        "S3.near_ood", "S3.seed_robustness", "S3.calibration", "S3.openset_detection", "S3.noise_grid",
        "S4.regime"]
    only_train = P.build_plan(REPO, to_step="T8")
    assert len(only_train) == 27 and all(p["kind"] == "train" for p in only_train)
    assert [p["id"] for p in P.build_plan(REPO, from_step="T8", to_step="T8")] == [f"T8.{i}" for i in range(1, 7)]


@pytest.mark.parametrize("kw,msg", [
    ({"stages": ["T9"]}, "알 수 없는 학습 단계"),
    ({"exp": "cifar100_softmax"}, "큐(run_all.ps1 T1~T8)에 없는"),           # 큐 밖, results/에 없는 런
    ({"exp": "diag224_nmfc_combo"}, "큐(run_all.ps1 T1~T8)에 없는"),         # 큐 밖, results/에 커밋된 진단 런
    ({"exp": "no_such_experiment"}, "어디에도 정의되지 않은"),
    ({"exp": "c100v_shrb", "stages": ["T5"]}, "함께 쓸 수 없다"),
    ({"from_step": "S9"}, "알 수 없는 단계 ID"),
    ({"from_step": "S4", "to_step": "S3"}, "실행할 항목이 없다"),
])
def test_invalid_requests_exit_2(capsys, repo_untouched, kw, msg):
    assert P.run_pipeline(REPO, dry_run=True, **kw) == 2
    assert msg in capsys.readouterr().out


def test_find_exp_refuses_runs_outside_queue():
    with pytest.raises(LookupError, match="configs/cifar100.yaml"):
        P.find_exp("cifar100_softmax", REPO)
    with pytest.raises(LookupError, match="configs/synthetic.yaml"):
        P.find_exp("syn_moons_apt_none", REPO)
    with pytest.raises(LookupError, match="configs/synthetic.yaml"):      # config에는 있으나 큐에서 뺀 런
        P.find_exp("syn_moons_apt_fixed3", REPO)


def test_standalone_main(capsys, repo_untouched):
    assert P.main(["--dry-run", "--stage", "T1"]) == 0
    assert "T1.1" in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:                  # 실제 실행은 --workdir 필수
        P.main(["--stage", "T1"])
    assert e.value.code == 2


# ---------------------------------------------------------------- 2) 허용 exit 정책·STEPS 정의

def test_allowed_exit_policy():
    S = P.STEP_BY_ID
    assert "S4.conditions" not in S and not [i for i in S if i.startswith("S5")]
    assert all(s["allowed_exit"] == [0] for s in P.STEPS)                     # exit 0만 허용
    assert P.exit_allowed({"allowed_exit": [0, 1]}, 1) and not P.exit_allowed({"allowed_exit": [0, 1]}, 2)
    assert not P.exit_allowed(S["S4.regime"], 1) and P.exit_allowed(S["S4.regime"], 0)
    assert not P.exit_allowed(S["S4.theory"], 1) and P.exit_allowed(S["S4.theory"], 0)
    assert not P.exit_allowed(S["S3.retrieval"], None)
    assert [s["id"] for s in P.STEPS if not s["required"]] == ["S2.ext"]      # 선택은 ext뿐
    assert S["S3.near_ood"]["needs_any"] == [f"checkpoints/{e}/ckpt.pt" for e in P.NEAR_OOD_EXPS]
    assert len(P.NEAR_OOD_EXPS) == 15
    assert len(S["S3.calibration"]["needs"]) == 5
    assert S["S3.openset_detection"]["needs"] == ["checkpoints/c100v_softmax_open/ckpt.pt"]
    assert S["S1.bank_shrb"]["needs"] == ["checkpoints/c100v_shrb_open/ckpt.pt"]
    for sid in ("S3.retrieval", "S3.fewshot_verify", "S3.stats_ci", "S3.seed_robustness", "S3.noise_grid",
                "S4.regime", "S4.theory"):
        assert not S[sid]["needs"] and not S[sid]["needs_any"], sid
    assert {"expect": "통과 16/16", "min": 1} in S["S4.theory"]["post"]


def test_steps_follow_ps1_except_documented_changes():
    text = (REPO / P.PS1_REL).read_text(encoding="utf-8")
    calls = P.parse_ps1_scripts(text)
    assert len(calls) == 13                               # :55-56, :59, :62-69, :72-73
    want = list(calls)
    i, j = want.index(("scripts/eval_seed_robustness.py", ())), want.index(("scripts/eval_near_ood.py", ()))
    assert j == i + 1                                     # ps1:65-66은 seed_robustness가 먼저(결함)
    want[i], want[j] = want[j], want[i]
    port = []
    for s in P.STEPS:
        if s["ps1"] is None:
            continue
        if s["id"] == "S2.ext":                           # resnet50만 부르는 래퍼로 대체
            port.append(("scripts/embed_external.py", ()))
        else:
            port.append((s["cmd"][0], tuple(s["cmd"][1:])))
    assert port == want
    lines = text.split("\n")
    for s in P.STEPS:
        if s["ps1"] and s["id"] != "S2.ext":
            assert Path(s["cmd"][0]).name in lines[int(s["ps1"]) - 1].replace("\\", "/"), s["id"]
    assert [s["id"] for s in P.STEPS if s["ps1"] is None] == []
    assert [s["id"] for s in P.STEPS if not (REPO / s["cmd"][0]).is_file()] == []   # 모든 단계 스크립트가 실재


def _dotted(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        return _dotted(node.func) + "()"
    if isinstance(node, ast.Subscript):
        return _dotted(node.value) + "[]"
    return "?"


def _module_level_calls(path: Path) -> tuple[set[str], ast.If | None]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls: set[str] = set()
    main_block = None
    for st in tree.body:
        if isinstance(st, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant):
            continue                                       # docstring
        if isinstance(st, (ast.FunctionDef, ast.ClassDef)):
            for d in st.decorator_list:                    # 데코레이터는 import 때 평가된다
                calls |= {_dotted(c.func) for c in ast.walk(d) if isinstance(c, ast.Call)}
            continue
        if isinstance(st, ast.If) and "__name__" in ast.unparse(st.test):
            main_block = st
            continue
        assert isinstance(st, (ast.Assign, ast.Expr)), f"{path.name}:{st.lineno} 예상 밖 모듈 수준 문장"
        calls |= {_dotted(c.func) for c in ast.walk(st) if isinstance(c, ast.Call)}
    return calls, main_block


def test_embed_external_import_is_side_effect_free():
    """S2 래퍼가 import해도 되는지: 모듈 수준 호출은 경로 계산·장치 조회·변환 객체 생성뿐이어야 한다."""
    ee = REPO / "scripts" / "embed_external.py"
    calls, main_block = _module_level_calls(ee)
    assert calls <= {"sys.path.insert", "str", "Path", "Path().resolve", "get_device", "torch.no_grad",
                     "transforms.Compose", "transforms.Resize", "transforms.CenterCrop",
                     "transforms.ToTensor", "transforms.Normalize"}, calls
    first = main_block.body[1].value                       # body[0]은 print(device)
    assert ast.unparse(first) == "run('ext_resnet50_frozen', m_resnet50)"
    funcs = {n.name: n for n in ast.parse(ee.read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef)}
    assert {"run", "m_resnet50", "save", "embed", "cifar_test"} <= set(funcs)
    assert any(isinstance(n, ast.ExceptHandler) for n in ast.walk(funcs["run"]))   # 예외를 삼킨다 → mtime 판정
    kw = {k.arg: k.value for c in ast.walk(funcs["cifar_test"]) if isinstance(c, ast.Call) for k in c.keywords}
    assert isinstance(kw["download"], ast.Constant) and kw["download"].value is False  # 데이터 사전조건 근거
    for dep in ("src/evaluate.py", "src/utils/misc.py"):    # embed_external이 import하는 저장소 모듈
        dep_calls, _ = _module_level_calls(REPO / dep)
        assert dep_calls <= {"Path", "Path().resolve", "torch.no_grad"}, (dep, dep_calls)


# ---------------------------------------------------------------- 3) 스텁 저장소에서 실제 실행

STUB_TRAIN = r'''"""가짜 src.train — run_or_load 캐시 계약만 흉내 낸다(학습 없음)."""
import argparse, json, os, sys
from pathlib import Path
import yaml
ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--only", nargs="*", default=None)
ap.add_argument("--force", action="store_true")
a = ap.parse_args()
p = Path(a.config)
cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
if "base" in cfg:
    cfg = {**yaml.safe_load((p.parent / cfg["base"]).read_text(encoding="utf-8")), **cfg}
fail = set(filter(None, os.environ.get("STUB_TRAIN_FAIL", "").split(",")))
noop = set(filter(None, os.environ.get("STUB_TRAIN_NOOP", "").split(",")))
npz = set(filter(None, os.environ.get("STUB_TRAIN_NPZ", "").split(",")))
for e in cfg["experiments"]:
    name = {**cfg.get("defaults", {}), **e}["name"]
    if a.only and name not in a.only:
        continue
    f = Path("results") / name / "metrics.json"
    if f.exists() and not a.force:
        print(name, "캐시")
        continue
    if name in noop:                      # 아무것도 안 하고 '성공'한 척(공허 합격 모사)
        print(name, "건너뜀(스텁)")
        continue
    if f.exists():
        f.unlink()
    if name in fail:
        print(name, "학습 실패(스텁)", flush=True)
        sys.exit(3)
    f.parent.mkdir(parents=True, exist_ok=True)
    if name in npz:                       # 실제 학습처럼 embeddings.npz를 먼저, metrics.json(완료 센티널)을 마지막에
        (f.parent / "embeddings.npz").write_bytes(b"npz " + name.encode())
    f.write_text(json.dumps({"name": name}), encoding="utf-8")
    print(name, "학습 완료(스텁)")
'''

STUB_EMBED_OK = r'''from pathlib import Path
device = "cpu"
def m_resnet50():
    return "z", "y"
def run(name, fn):
    try:
        fn()
        d = Path("results") / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "embeddings.npz").write_bytes(b"npz")
        print(f"[OK] {name}")
    except Exception as e:
        print(f"[SKIP] {name}: {type(e).__name__}: {e}")
'''
STUB_EMBED_SKIP = STUB_EMBED_OK.replace('return "z", "y"', 'raise RuntimeError("데이터 없음(스텁)")')


def make_wd(tmp_path: Path) -> Path:
    wd = tmp_path / "wd"
    shutil.copytree(REPO / "configs", wd / "configs")
    (wd / "scripts").mkdir(parents=True)
    shutil.copy2(REPO / P.PS1_REL, wd / P.PS1_REL)
    (wd / "src").mkdir()
    (wd / "src" / "__init__.py").write_text("", encoding="utf-8")
    (wd / "src" / "train.py").write_text(STUB_TRAIN, encoding="utf-8")
    return wd


def stub(wd: Path, rel: str, *, writes=(), prints=(), code: int = 0, source: str | None = None) -> None:
    p = wd / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if source is None:
        body = ["import sys", "from pathlib import Path"]
        for path, content in writes:
            body.append(f"q = Path({path!r}); q.parent.mkdir(parents=True, exist_ok=True); "
                        f"q.write_text({content!r}, encoding='utf-8')")
        body += [f"print({s!r})" for s in prints] + [f"sys.exit({code})"]
        source = "\n".join(body) + "\n"
    p.write_text(source, encoding="utf-8")


def touch(path: Path, data: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def status_of(wd: Path) -> dict:
    return json.loads((wd / P.STATUS_REL).read_text(encoding="utf-8"))


def item(st: dict, item_id: str) -> dict:
    return next(r for r in st["items"] if r["id"] == item_id)


def test_stub_stage_T1_force_then_cache(tmp_path):
    wd = make_wd(tmp_path)
    assert P.run_pipeline(wd, stages=["T1"], force=True) == 0
    st = status_of(wd)
    assert st["pass"] is True and st["exit_code"] == 0 and st["queue_parity"]["ok"] is True
    assert st["trained"] == T1_CONFIG_ORDER                                   # 9런, config 순서
    r = item(st, "T1.1")
    assert r["kind"] == "train" and r["exit"] == 0 and r["ok"] is True and r["skipped_reason"] is None
    assert r["cmd"][:2] == [sys.executable, "-B"] and r["cmd"][-1] == "--force"
    assert all(c["ok"] for c in r["post"]) and any("재학습 9런" in c["cond"] for c in r["post"])
    assert (wd / r["log"]).is_file() and any("학습 완료(스텁)" in ln for ln in r["output_tail"])
    for key in ("started", "finished", "args", "items", "trained", "pass"):          # 상태 JSON 필수 키
        assert key in st
    for key in ("id", "kind", "cmd", "exit", "ok", "duration_s", "trained", "post", "skipped_reason"):
        assert key in r
    # 두 번째: --force 없음 → 캐시라 학습 0런이 기대값이고 통과
    assert P.run_pipeline(wd, stages=["T1"]) == 0
    st2 = status_of(wd)
    assert st2["trained"] == [] and item(st2, "T1.1")["cached"] == T1_CONFIG_ORDER
    # 캐시 일부 삭제 → 그 런만 학습해야 한다
    (wd / "results" / "syn_swiss_nmfc" / "metrics.json").unlink()
    assert P.run_pipeline(wd, stages=["T1"]) == 0
    assert status_of(wd)["trained"] == ["syn_swiss_nmfc"]


def _old_file(path: Path, data: str = "{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")
    os.utime(path, ns=(1_600_000_000_000_000_000, 1_600_000_000_000_000_000))   # 2020-09: 커밋본 흉내


def test_stub_records_artifact_fingerprints(tmp_path, monkeypatch):
    """학습 항목은 이번에 학습한 런마다 metrics.json·embeddings.npz 지문을 남긴다(compare의 되돌리기 탐지 근거)."""
    wd = make_wd(tmp_path)
    _old_file(wd / "results" / "syn_moons_nmfc" / "embeddings.npz", "old npz")    # 학습이 다시 쓰지 않는 옛 npz
    monkeypatch.setenv("STUB_TRAIN_NPZ", "syn_moons_softmax")
    assert P.run_pipeline(wd, stages=["T1"], force=True) == 0
    arts = item(status_of(wd), "T1.1")["artifacts"]
    assert sorted(arts) == sorted(T1_CONFIG_ORDER)
    for e in T1_CONFIG_ORDER:
        mp = wd / "results" / e / "metrics.json"
        st = mp.stat()
        assert arts[e]["metrics.json"] == {"sha256": hashlib.sha256(mp.read_bytes()).hexdigest(),
                                           "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    npz = wd / "results" / "syn_moons_softmax" / "embeddings.npz"
    assert arts["syn_moons_softmax"]["embeddings.npz"]["sha256"] == hashlib.sha256(npz.read_bytes()).hexdigest()
    assert arts["syn_moons_softmax"]["embeddings.npz"]["rewritten"] is True
    assert arts["syn_moons_nmfc"]["embeddings.npz"]["rewritten"] is False
    assert arts["syn_circles_nmfc"]["embeddings.npz"] is None
    monkeypatch.delenv("STUB_TRAIN_NPZ")
    assert P.run_pipeline(wd, stages=["T1"]) == 0                              # 캐시: 학습 0런 → 지문 없음
    assert item(status_of(wd), "T1.1")["artifacts"] == {}


def test_stub_force_over_committed_metrics_counts_rewrites(tmp_path, monkeypatch):
    """`train --stage T1 --force`: metrics.json이 이미 있는 볼륨에서 '다시 쓴 런 수 = 9'여야 한다."""
    wd = make_wd(tmp_path)
    for n in T1_CONFIG_ORDER:
        _old_file(wd / "results" / n / "metrics.json")
    assert P.run_pipeline(wd, stages=["T1"]) == 0                              # --force 없음: 전부 캐시
    assert status_of(wd)["trained"] == []
    assert P.run_pipeline(wd, stages=["T1"], force=True) == 0
    assert status_of(wd)["trained"] == T1_CONFIG_ORDER
    for n in T1_CONFIG_ORDER:
        _old_file(wd / "results" / n / "metrics.json")
    monkeypatch.setenv("STUB_TRAIN_NOOP", "syn_circles_nmfc")                  # exit 0인데 한 런을 안 씀
    assert P.run_pipeline(wd, stages=["T1"], force=True) == 1
    r = item(status_of(wd), "T1.1")
    assert r["exit"] == 0 and r["ok"] is False and "syn_circles_nmfc" not in r["trained"]
    assert len(r["trained"]) == 8


def test_stub_fresh_detects_stale_existing_output(tmp_path):
    wd = make_wd(tmp_path)
    _old_file(wd / "results" / "retrieval_eval.json", "[]")                    # 커밋된 옛 산출물
    stub(wd, "scripts/eval_retrieval.py", prints=["옛 파일을 그대로 둠"])
    assert P.run_pipeline(wd, from_step="S3.retrieval", to_step="S3.retrieval") == 1
    r = item(status_of(wd), "S3.retrieval")
    assert [c["detail"] for c in r["post"] if c["cond"].startswith("갱신됨")] == ["mtime 그대로"]
    stub(wd, "scripts/eval_retrieval.py", writes=[("results/retrieval_eval.json", "[1]")])
    assert P.run_pipeline(wd, from_step="S3.retrieval", to_step="S3.retrieval") == 0


def test_stub_exp_then_aggregate_runs_four_and_orders_near_ood(tmp_path):
    """`train --exp c100v_shrb --force --then aggregate`: 1런 + 집계 4종(ckpt 없으면 near_ood 건너뜀)."""
    wd = make_wd(tmp_path)
    for rel, out in (("eval_retrieval", "retrieval_eval.json"), ("eval_fewshot_verify", "fewshot_verify_eval.json"),
                     ("eval_stats_ci", "stats_ci.json"), ("eval_seed_robustness", "seed_robustness.json")):
        stub(wd, f"scripts/{rel}.py", writes=[(f"results/{out}", "{}")])
    assert P.run_pipeline(wd, exp="c100v_shrb", force=True, then_aggregate=True) == 0
    st = status_of(wd)
    assert [r["id"] for r in st["items"]] == ["T5:c100v_shrb"] + AGG
    assert [r["exit"] for r in st["items"]] == [0, 0, 0, 0, None, 0]
    assert item(st, "S3.near_ood")["skipped_reason"].startswith("사전조건 미충족")
    assert st["trained"] == ["c100v_shrb"] and st["pass"] is True


def test_stub_train_failure_continues_training_but_blocks_followups(tmp_path, monkeypatch):
    wd = make_wd(tmp_path)
    monkeypatch.setenv("STUB_TRAIN_FAIL", "syn_moons_nmfc")
    assert P.run_pipeline(wd, stages=["T1", "T3"], force=True, then_aggregate=True) == 1
    st = status_of(wd)
    t1 = item(st, "T1.1")
    assert t1["exit"] == 3 and t1["ok"] is False
    assert t1["trained"] == T1_CONFIG_ORDER[:4]                               # 실패 전까지 4런
    assert item(st, "T3.1")["ok"] is True                                     # 다음 학습은 계속
    for sid in AGG:
        assert item(st, sid)["skipped_reason"].startswith("학습 실패")
        assert item(st, sid)["exit"] is None
    assert st["pass"] is False and st["exit_code"] == 1


def test_stub_exp_mode_trains_exactly_one(tmp_path):
    wd = make_wd(tmp_path)
    assert P.run_pipeline(wd, exp="c100v_shrb", force=True) == 0
    st = status_of(wd)
    assert st["trained"] == ["c100v_shrb"]
    assert item(st, "T5:c100v_shrb")["cmd"][2:] == [
        "-m", "src.train", "--config", "configs/cifar100_variants.yaml", "--only", "c100v_shrb", "--force"]
    assert not (wd / "results" / "c100v_mhrb").exists()                       # 같은 config의 다른 런은 안 건드림


def _audit_stubs(wd: Path, regime_code: int = 0, theory_line: str = "통과 16/16   R=5.756") -> None:
    stub(wd, "scripts/audit_operating_regime.py", writes=[("results/operating_regime_audit.json", "{}")],
         prints=["통과 3/8"], code=regime_code)
    stub(wd, "scripts/audit_theory_claims.py", writes=[("results/theory_audit.json", "{}")],
         prints=[theory_line])


def test_stub_audit_group_runs_regime_then_theory(tmp_path):
    wd = make_wd(tmp_path)
    _audit_stubs(wd)
    assert P.run_pipeline(wd, from_step="S4", to_step="S4") == 0
    st = status_of(wd)
    assert [r["id"] for r in st["items"]] == ["S4.regime", "S4.theory"]
    assert all(r["exit"] == 0 and r["ok"] is True for r in st["items"])


def test_stub_disallowed_exit_stops_the_chain(tmp_path):
    wd = make_wd(tmp_path)
    _audit_stubs(wd, regime_code=1)                       # 허용 exit는 0뿐이다
    assert P.run_pipeline(wd, from_step="S4", to_step="S4") == 1
    st = status_of(wd)
    assert item(st, "S4.regime")["exit"] == 1 and item(st, "S4.regime")["ok"] is False
    assert item(st, "S4.theory")["skipped_reason"] == "앞선 필수 단계 S4.regime 실패로 중단"


def test_stub_missing_expected_string_fails(tmp_path):
    wd = make_wd(tmp_path)
    _audit_stubs(wd, theory_line="통과 15/16   R=5.756")
    assert P.run_pipeline(wd, from_step="S4", to_step="S4") == 1
    r = item(status_of(wd), "S4.theory")
    assert r["exit"] == 0 and r["ok"] is False
    assert [c["cond"] for c in r["post"] if not c["ok"]] == ["출력에 '통과 16/16' 1회 이상"]


def test_stub_postcondition_fresh_is_required(tmp_path):
    wd = make_wd(tmp_path)
    stub(wd, "scripts/eval_retrieval.py", prints=["아무것도 안 씀"])            # exit 0인데 산출물 없음
    assert P.run_pipeline(wd, from_step="S3.retrieval", to_step="S3.retrieval") == 1
    r = item(status_of(wd), "S3.retrieval")
    assert r["exit"] == 0 and r["ok"] is False
    stub(wd, "scripts/eval_retrieval.py", writes=[("results/retrieval_eval.json", "[]")])
    assert P.run_pipeline(wd, from_step="S3.retrieval", to_step="S3.retrieval") == 0


def test_stub_near_ood_runs_only_with_ckpt_and_needs_15_rows(tmp_path):
    wd = make_wd(tmp_path)
    rows15 = json.dumps({e: {"near_auc": 0.5} for e in P.NEAR_OOD_EXPS})
    stub(wd, "scripts/eval_near_ood.py", writes=[("results/near_ood.json", rows15)])
    assert P.run_pipeline(wd, from_step="S3.near_ood", to_step="S3.near_ood") == 0
    r = item(status_of(wd), "S3.near_ood")
    assert r["exit"] is None and r["skipped_reason"].startswith("사전조건 미충족")
    assert not (wd / "results" / "near_ood.json").exists()                     # 실행하지 않았다
    touch(wd / "checkpoints" / "ood10v_shrb_s2" / "ckpt.pt")                   # 15개 중 1개만 있어도 실행
    assert P.run_pipeline(wd, from_step="S3.near_ood", to_step="S3.near_ood") == 0
    assert item(status_of(wd), "S3.near_ood")["exit"] == 0
    rows14 = json.dumps({e: {"near_auc": 0.5} for e in P.NEAR_OOD_EXPS[:14]})
    stub(wd, "scripts/eval_near_ood.py", writes=[("results/near_ood.json", rows14)])
    assert P.run_pipeline(wd, from_step="S3.near_ood", to_step="S3.near_ood") == 1


def test_stub_calibration_needs_all_five_ckpts(tmp_path):
    wd = make_wd(tmp_path)
    stub(wd, "scripts/eval_calibration.py", writes=[("results/calibration_eval.json", "{}")])
    for e in P.CALIBRATION_CKPT_EXPS[:4]:
        touch(wd / "checkpoints" / e / "ckpt.pt")
    assert P.run_pipeline(wd, from_step="S3.calibration", to_step="S3.calibration") == 0
    assert "cifar100_deepmfa" in item(status_of(wd), "S3.calibration")["skipped_reason"]
    touch(wd / "checkpoints" / "cifar100_deepmfa" / "ckpt.pt")
    assert P.run_pipeline(wd, from_step="S3.calibration", to_step="S3.calibration") == 0
    assert item(status_of(wd), "S3.calibration")["exit"] == 0


def test_stub_runtime_queue_parity_guard(tmp_path):
    wd = make_wd(tmp_path)
    ps1 = wd / P.PS1_REL
    lines = ps1.read_text(encoding="utf-8").split("\n")
    assert "ood_cifar10_softmax_lr3" in lines[36 - 1]
    del lines[36 - 1]                                                          # ood_cifar10_softmax_lr3 줄
    ps1.write_text("\n".join(lines), encoding="utf-8")
    assert P.run_pipeline(wd, stages=["T1"], force=True) == 2
    st = status_of(wd)
    assert st["exit_code"] == 2 and st["queue_parity"]["ok"] is False and st["items"] == []
    assert not (wd / "results").exists()                                       # 아무것도 실행하지 않았다


def test_stub_out_export(tmp_path):
    wd = make_wd(tmp_path)
    out = tmp_path / "out"
    assert P.run_pipeline(wd, stages=["T1"], force=True, out=out) == 0
    exported = json.loads((out / "reproduce" / "pipeline_status.json").read_text(encoding="utf-8"))
    assert exported["pass"] is True and exported["trained"] == T1_CONFIG_ORDER
    assert exported["export"]["files"] == ["pipeline_status.json", "pipeline_logs/"]
    assert (out / "reproduce" / "pipeline_logs").is_dir()


def test_stub_embed_wrapper_judges_by_npz_mtime(tmp_path):
    wd = make_wd(tmp_path)
    run = [sys.executable, "-B", str(P._SELF), "--embed-resnet50"]
    stub(wd, "scripts/embed_external.py", source=STUB_EMBED_OK)
    r = subprocess.run(run, cwd=wd, env=P.child_env(), capture_output=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "[device] cpu" in r.stdout and (wd / P.EXT_NPZ).is_file()
    stub(wd, "scripts/embed_external.py", source=STUB_EMBED_SKIP)               # run()이 예외를 삼킨다
    r = subprocess.run(run, cwd=wd, env=P.child_env(), capture_output=True, encoding="utf-8", errors="replace")
    assert r.returncode == 1 and "[SKIP] ext_resnet50_frozen" in r.stdout and "[FAIL]" in r.stdout


def test_stub_ext_step_with_and_without_flag(tmp_path, monkeypatch):
    wd = make_wd(tmp_path)
    stub(wd, "scripts/embed_external.py", source=STUB_EMBED_OK)
    assert P.run_pipeline(wd, from_step="S2", to_step="S2") == 0               # --with-ext 아님 → 건너뜀
    assert item(status_of(wd), "S2.ext")["skipped_reason"].startswith("--with-ext 미지정")
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))
    assert P.run_pipeline(wd, with_ext=True, from_step="S2", to_step="S2") == 0
    assert "resnet50-11ad3fa6.pth" in item(status_of(wd), "S2.ext")["skipped_reason"]
    touch(tmp_path / "torch" / "hub" / "checkpoints" / "resnet50-11ad3fa6.pth")
    touch(wd / "data" / "raw" / "cifar-100-python" / "test")
    touch(wd / "data" / "raw" / "cifar-100-python" / "meta")
    assert P.run_pipeline(wd, with_ext=True, from_step="S2", to_step="S2") == 0   # train 없음 → 건너뜀
    assert "cifar-100-python/train" in item(status_of(wd), "S2.ext")["skipped_reason"]
    touch(wd / "data" / "raw" / "cifar-100-python" / "train")                 # torchvision 무결성 검사가 train도 본다
    assert P.run_pipeline(wd, with_ext=True, from_step="S2", to_step="S2") == 0
    r = item(status_of(wd), "S2.ext")
    assert r["exit"] == 0 and r["ok"] is True and r["required"] is False
    stub(wd, "scripts/embed_external.py", source=STUB_EMBED_SKIP)               # 선택 단계 실패는 전체 통과
    assert P.run_pipeline(wd, with_ext=True, from_step="S2", to_step="S2") == 0
    assert item(status_of(wd), "S2.ext")["ok"] is False
