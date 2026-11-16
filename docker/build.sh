#!/usr/bin/env bash
# docker/build.sh — nmfc:cpu·nmfc:gpu의 유일한 빌드 경로(Linux·CI). docker/build.ps1과 같은 일을 한다.
#
# 사용: bash docker/build.sh <cpu|gpu> [--skip-build]
#   1) 내보내기: ${NMFC_CTX:-${TMPDIR:-/tmp}/nmfc-ctx}를 비우고 git checkout-index -a -f --prefix=<ctx>/
#      (인덱스가 비어 있으면 HEAD — git archive HEAD와 같은 내용을 임시 인덱스의 read-tree로 내보낸다)
#      리비전: git rev-parse --short=12 HEAD. 내보낸 인덱스가 HEAD와 다르면(git diff --cached --quiet 실패)
#      '+index'를 붙인다(인덱스가 비어 HEAD를 내보냈으면 붙이지 않는다). HEAD가 없으면 no-commit.
#   2) lock:  $PYTHON docker/nmfc_cli.py manifest --write-lock <ctx>   (내보낸 사본을 cwd=<ctx>에서 실행)
#   3) 해시:  NMFC_MANIFEST_SHA256 = sha256(<ctx>/docker/context.lock)
#   4) 빌드:  docker build --provenance=false --target <t> --build-arg NMFC_MANIFEST_SHA256=…
#             --build-arg NMFC_MANIFEST_SOURCE=host-export --build-arg NMFC_REVISION=<리비전> -t nmfc:<t> <ctx>
#   5) 기록:  docker image inspect → out/build_<t>.json (Id, RootFS.Layers, Size, Labels 등)
#   --skip-build는 1~3만 한다(데몬 불필요, 시험용).
# 환경변수: PYTHON(기본 python3, 표준 라이브러리만 사용), DOCKER(기본 docker), NMFC_CTX(내보낼 ASCII 경로)
set -euo pipefail
IFS=$'\n\t'

log() { printf '[build.sh] %s\n' "$*"; }
die() { printf '[build.sh] 실패: %s\n' "$*" >&2; exit 1; }
usage() { printf 'usage: bash docker/build.sh <cpu|gpu> [--skip-build]\n' >&2; exit 2; }

[[ $# -ge 1 && $# -le 2 ]] || usage
target=$1
[[ $target == cpu || $target == gpu ]] || usage
skip_build=0
if [[ $# -eq 2 ]]; then
  [[ $2 == --skip-build ]] || usage
  skip_build=1
fi

PYTHON=${PYTHON:-python3}
DOCKER=${DOCKER:-docker}

sha256_file() {
  local out
  if command -v sha256sum > /dev/null 2>&1; then
    out=$(sha256sum -- "$1")
  else
    out=$(shasum -a 256 -- "$1")
  fi
  printf '%s\n' "${out%% *}"
}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_root=$(cd -- "$script_dir/.." && pwd -P)
[[ -f $repo_root/docker/finalize.sh ]] || die "저장소 루트가 아니다(docker/finalize.sh 없음): $repo_root"
git -C "$repo_root" rev-parse --is-inside-work-tree > /dev/null || die "git 작업 트리가 아니다: $repo_root"

# ------------------------------------------------------------------ 컨텍스트 경로 안전장치
ctx=${NMFC_CTX:-${TMPDIR:-/tmp}/nmfc-ctx}
ctx=${ctx%/}
ctx_name=$(basename -- "$ctx")
[[ $ctx_name == nmfc-ctx* ]] || die "컨텍스트 디렉터리 이름은 nmfc-ctx로 시작해야 한다(지우고 다시 만든다): $ctx"
if LC_ALL=C grep -q '[^ -~]' <<< "$ctx"; then
  die "컨텍스트 경로는 ASCII여야 한다(구버전 BuildKit은 비ASCII 컨텍스트에서 실패한다): $ctx"
fi
ctx_parent=$(cd -- "$(dirname -- "$ctx")" 2> /dev/null && pwd -P) || die "컨텍스트의 부모 디렉터리가 없다: $ctx"
ctx="$ctx_parent/$ctx_name"
case "$ctx/" in
  "$repo_root"/*) die "컨텍스트는 저장소 밖이어야 한다: $ctx" ;;
esac

# ------------------------------------------------------------------ 1) 내보내기
rm -rf -- "$ctx"
mkdir -p -- "$ctx"
n_index=$(git -C "$repo_root" ls-files | wc -l)
n_index=${n_index//[[:space:]]/}
if [[ $n_index -gt 0 ]]; then
  log "내보내기: git checkout-index -a -f (인덱스 ${n_index}개) → $ctx"
  git -C "$repo_root" checkout-index -a -f --prefix="$ctx/"
else
  # 인덱스가 비면 HEAD를 내보낸다('git archive HEAD'와 같은 내용). 임시 인덱스 파일을 쓰므로 실제 인덱스는
  # 건드리지 않는다. build.ps1과 같은 방법이다(Windows tar.exe는 한글 경로를 풀지 못해 git archive|tar를 쓰지 않는다).
  log "인덱스가 비어 있다 — 임시 인덱스(git read-tree HEAD)로 HEAD를 내보낸다"
  tmp_index=$ctx_parent/nmfc-ctx-index-$$
  rm -f -- "$tmp_index"
  trap 'rm -f -- "$tmp_index"' EXIT
  GIT_INDEX_FILE=$tmp_index git -C "$repo_root" read-tree HEAD
  GIT_INDEX_FILE=$tmp_index git -C "$repo_root" checkout-index -a -f --prefix="$ctx/"
  rm -f -- "$tmp_index"
fi
# 내보낸 트리에 없는 것만 센다: 스테이징하지 않은 변경(작업트리 대 인덱스)과 미추적·비무시 파일.
# 스테이징된 항목(인덱스 대 HEAD)은 내보내기에 들어가므로 세지 않는다(status --porcelain 줄 수는
# 'git add -A' 직후에도 거짓 경고를 냈다).
n_unstaged=$(git -C "$repo_root" --no-optional-locks diff --name-only | wc -l)
n_unstaged=${n_unstaged//[[:space:]]/}
n_untracked=$(git -C "$repo_root" --no-optional-locks ls-files -o --exclude-standard | wc -l)
n_untracked=${n_untracked//[[:space:]]/}
if [[ $((n_unstaged + n_untracked)) -gt 0 ]]; then
  log "경고: 내보낸 트리에 없는 변경 — unstaged ${n_unstaged}건, untracked ${n_untracked}건. 이미지는 git 인덱스(비었으면 HEAD) 내용으로 빌드된다(넣으려면 먼저 git add -A)"
else
  log "작업트리 = 인덱스(unstaged·untracked 0건)"
fi
# 리비전 라벨(org.opencontainers.image.revision)은 내보낸 내용을 따른다: 인덱스를 내보냈고 그것이 HEAD와 다르면
# '+index'를 붙이고, 인덱스가 비어 HEAD를 내보냈으면 붙이지 않는다. HEAD가 없으면(커밋 전) no-commit.
if revision=$(git -C "$repo_root" rev-parse --verify --quiet --short=12 HEAD); then
  if [[ $n_index -gt 0 ]]; then
    rc=0
    git -C "$repo_root" --no-optional-locks diff --cached --quiet || rc=$?
    case $rc in
      0) ;;
      1) revision="$revision+index" ;;
      *) die "git diff --cached --quiet 실패(exit $rc)" ;;
    esac
  fi
else
  revision=no-commit
fi
log "리비전: NMFC_REVISION=$revision"

# ------------------------------------------------------------------ 2) lock
log "lock: $PYTHON docker/nmfc_cli.py manifest --write-lock $ctx"
(cd -- "$ctx" && PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 "$PYTHON" -B docker/nmfc_cli.py manifest --write-lock "$ctx")
lock=$ctx/docker/context.lock
files=$ctx/docker/manifest.files
[[ -s $lock ]] || die "manifest --write-lock가 $lock을 만들지 않았다"
[[ -s $files ]] || die "manifest --write-lock가 $files를 만들지 않았다"
# grep은 Windows(MSYS)에서 줄 끝 CR을 떼고 보므로 tr로 센다.
n_cr=$(LC_ALL=C tr -cd '\r' < "$lock" | wc -c)
n_cr=${n_cr//[[:space:]]/}
if [[ $n_cr -gt 0 ]]; then
  die "docker/context.lock에 CR이 ${n_cr}개 있다 — LF로 써야 sha256이 플랫폼과 무관하게 같다"
fi

# ------------------------------------------------------------------ 3) 해시
sha=$(sha256_file "$lock")
n_files=$(wc -l < "$files")
n_files=${n_files//[[:space:]]/}
log "NMFC_MANIFEST_SHA256=$sha (매니페스트 파일 ${n_files}개)"

if [[ $skip_build -eq 1 ]]; then
  log "--skip-build: docker build 전에 멈춘다"
  exit 0
fi

# ------------------------------------------------------------------ 4) 빌드
tag="nmfc:$target"
"$DOCKER" build --provenance=false --progress=plain --target "$target" \
  --build-arg "NMFC_MANIFEST_SHA256=$sha" \
  --build-arg "NMFC_MANIFEST_SOURCE=host-export" \
  --build-arg "NMFC_REVISION=$revision" \
  -t "$tag" "$ctx"

# ------------------------------------------------------------------ 5) 기록
mkdir -p -- "$repo_root/out"
out_json=$repo_root/out/build_$target.json
work=$(mktemp -d "${TMPDIR:-/tmp}/nmfc-build-record.XXXXXX")
trap 'rm -rf -- "$work"' EXIT
"$DOCKER" image inspect "$tag" > "$work/inspect.json"
server_version=$("$DOCKER" version --format '{{.Server.Version}}' 2> /dev/null || true)
storage_driver=$("$DOCKER" info --format '{{.Driver}}' 2> /dev/null || true)
driver_status=$("$DOCKER" info --format '{{json .DriverStatus}}' 2> /dev/null || true)
backend=cpu
if [[ $target == gpu ]]; then
  backend=cu128
fi
"$PYTHON" -B - "$work/inspect.json" "$out_json" "$target" "$tag" "$sha" "$n_files" "$ctx" \
  "$server_version" "$storage_driver" "$driver_status" "$backend" "$revision" <<'PY'
import datetime
import json
import sys

(inspect_path, out_path, target, tag, sha, n_files, ctx,
 server_version, storage_driver, driver_status, backend, revision) = sys.argv[1:13]
with open(inspect_path, encoding="utf-8") as f:
    img = json.load(f)[0]
labels = (img.get("Config") or {}).get("Labels") or {}
problems = []
if labels.get("nmfc.manifest.sha256") != sha:
    problems.append("nmfc.manifest.sha256 라벨이 컨텍스트 lock과 다르다")
if labels.get("nmfc.manifest.source") != "host-export":
    problems.append("nmfc.manifest.source 라벨이 host-export가 아니다")
if labels.get("nmfc.backend") != backend:
    problems.append(f"nmfc.backend 라벨이 {backend}가 아니다")
if labels.get("org.opencontainers.image.revision") != revision:
    problems.append(f"org.opencontainers.image.revision 라벨이 빌드 인자 NMFC_REVISION({revision})과 다르다")
if problems:
    sys.exit("[build.sh] 실패: " + "; ".join(problems))
record = {
    "target": target,
    "tag": tag,
    "id": img.get("Id"),
    "rootfs_layers": (img.get("RootFS") or {}).get("Layers", []),
    "size": img.get("Size"),
    "labels": labels,
    "created": img.get("Created"),
    "architecture": img.get("Architecture"),
    "os": img.get("Os"),
    "manifest_sha256": sha,
    "manifest_source": "host-export",
    "manifest_files": int(n_files),
    "context": ctx,
    "docker_server_version": server_version,
    "storage_driver": storage_driver,
    "driver_status": driver_status,
    "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
with open(out_path, "w", encoding="utf-8", newline="\n") as f:
    json.dump(record, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(f"[build.sh] {tag} {record['id']} ({len(record['rootfs_layers'])} layers, {record['size']} bytes)")
print(f"[build.sh] 기록 → {out_path}")
PY
