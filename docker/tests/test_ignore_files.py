""".gitignore ↔ .dockerignore 불변식 (.dockerignore 머리 주석).

`git add -A`가 인덱스에 넣는 파일(= git ls-files -co --exclude-standard)은 내보내기·context.lock에 들어간다.
그 파일을 .dockerignore가 빼면 클론에서 직접 빌드할 때 빌드 중 `manifest --verify --strict`가 missing으로 멈춘다.
반대로 이미지에 넣지 않을 폴더(data/·host-data/ 등)는 .gitignore에도 있어야 인덱스·lock에 들어가지 않는다
(compose 기본 바인드 원본 ./host-data가 .dockerignore에만 있었다).

.dockerignore 해석은 moby/patternmatcher(MatchesOrParentMatches)를 옮긴 것이다: 줄마다 공백 제거·'#' 주석·'!' 예외,
filepath.Clean 뒤 맨 앞 '/' 제거, '**'는 경로 성분 0개 이상('**/'의 '/'까지 먹음), '*'·'?'는 '/' 밖 문자,
뒤 패턴이 앞 판정을 뒤집고, 파일의 상위 디렉터리가 패턴과 맞아도 제외된다. 데몬 빌드 결과가 아니라 의미 재현이다.
"""
from __future__ import annotations

import posixpath
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

import nmfc_manifest as M  # noqa: E402

IS_GIT_REPO = shutil.which("git") is not None and (REPO / ".git").exists()
NOT_IN_IMAGE_DIRS = ("data", "checkpoints", "runs", "out", "dist", "_planning", ".venv", "host-data")


def _compile(pat: str):
    """moby patternmatcher compile(): (kind, 값). kind ∈ exact·prefix·suffix·regexp."""
    reg, kind, i, n = "^", "exact", 0, len(pat)
    idx = 0
    while i < n:
        ch = pat[i]
        if ch == "*":
            if i + 1 < n and pat[i + 1] == "*":
                i += 1
                if i + 1 < n and pat[i + 1] == "/":
                    i += 1
                if i + 1 >= n:
                    if kind == "exact":
                        kind = "prefix"
                    else:
                        reg += ".*"
                        kind = "regexp"
                else:
                    reg += "(.*/)?"
                    kind = "regexp"
                if idx == 0:
                    kind = "suffix"
            else:
                reg += "[^/]*"
                kind = "regexp"
        elif ch == "?":
            reg += "[^/]"
            kind = "regexp"
        elif ch in "[]":
            reg += ch
            kind = "regexp"
        else:
            reg += re.escape(ch)
        i += 1
        idx += 1
    return (kind, re.compile(reg + "$")) if kind == "regexp" else (kind, pat)


def _match(compiled, pat: str, path: str) -> bool:
    kind, val = compiled
    if kind == "exact":
        return path == pat
    if kind == "prefix":
        return path.startswith(pat[:-2])
    if kind == "suffix":
        suffix = pat[2:]
        return path.endswith(suffix) or (suffix.startswith("/") and path == suffix[1:])
    return bool(val.match(path))


def load_dockerignore(text: str) -> list[tuple[bool, str, tuple]]:
    pats = []
    for raw in text.lstrip("﻿").splitlines():
        if raw.startswith("#"):
            continue
        p = raw.strip()
        if not p:
            continue
        neg = p.startswith("!")
        if neg:
            p = p[1:].strip()
        if not p:
            continue
        p = posixpath.normpath(p)
        if len(p) > 1 and p.startswith("/"):
            p = p[1:]
        pats.append((neg, p, _compile(p)))
    return pats


def dockerignored(pats, path: str) -> bool:
    matched = False
    parent = posixpath.dirname(path)
    parts = parent.split("/") if parent else []
    for neg, pat, comp in pats:
        if neg != matched:              # 이미 제외된 파일엔 '!'만, 아직 아니면 제외 패턴만 본다
            continue
        hit = _match(comp, pat, path) or any(_match(comp, pat, "/".join(parts[:i + 1])) for i in range(len(parts)))
        if hit:
            matched = not neg
    return matched


def test_emulator_semantics():
    pats = load_dockerignore((REPO / ".dockerignore").read_text(encoding="utf-8"))
    want = {
        "out/x": True, "data/raw/x": True, "host-data/a.tar.gz": True, "a/b/c.log": True, "c.log": True,
        "docker/tests/x.log": False,                                     # !docker/** 가 *.log를 되돌린다
        "docker/__pycache__/a.pyc": True,                                # 그 뒤 **/__pycache__가 다시 뺀다
        "results/ext_resnet50_frozen/metrics.json": False, "results/ext_resnet50_frozen/embeddings.npz": True,
        "src/data/x.py": False,                                          # data/는 최상위만
        "Dockerfile": False,
    }
    got = {k: dockerignored(pats, k) for k in want}
    assert got == want


@pytest.mark.skipif(not IS_GIT_REPO, reason="git 저장소가 아님(컨테이너 등)")
def test_no_index_bound_file_is_dockerignored():
    """git add -A가 인덱스에 넣을 파일 중 .dockerignore가 빼는 것이 0개여야 한다(빌드 중 strict 검증 missing 방지)."""
    files = M.git_file_list(REPO)
    pats = load_dockerignore((REPO / ".dockerignore").read_text(encoding="utf-8"))
    assert len(files) > 100
    assert [f for f in files if dockerignored(pats, f)] == []
    # 대조(망가뜨린 입력): 추적 파일을 빼는 패턴을 더하면 잡혀야 한다
    bad = pats + load_dockerignore("docker/*.json\n")
    assert "docker/assets.json" in [f for f in files if dockerignored(bad, f)]


@pytest.mark.skipif(not IS_GIT_REPO, reason="git 저장소가 아님(컨테이너 등)")
@pytest.mark.parametrize("d", NOT_IN_IMAGE_DIRS)
def test_not_in_image_dirs_are_gitignored(d):
    """이미지에 넣지 않는(.dockerignore) 최상위 폴더는 .gitignore에도 있어야 git add -A로 인덱스·lock에 섞이지 않는다."""
    probe = f"{d}/nmfc-probe.bin"
    pats = load_dockerignore((REPO / ".dockerignore").read_text(encoding="utf-8"))
    assert dockerignored(pats, probe), d
    r = subprocess.run(["git", "-C", str(REPO), "--no-optional-locks", "check-ignore", "-q", "--no-index", probe],
                       stdin=subprocess.DEVNULL, capture_output=True)
    assert r.returncode == 0, f"{d}/ 가 .gitignore에 없다"
