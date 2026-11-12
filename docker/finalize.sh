#!/usr/bin/env bash
# docker/finalize.sh — 이미지 빌드 중 마무리
#
# 사용: bash docker/finalize.sh <cpu|gpu> <root|user>
#   Dockerfile의 cpu·gpu 타깃이 같은 스크립트를 두 번 부른다.
#     RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv bash docker/finalize.sh <t> root   # USER 전환 전(uid 0)
#     USER nmfc
#     RUN bash docker/finalize.sh <t> user                                                        # uid 1000
#   단언이 하나라도 실패하면 0이 아닌 코드로 끝나고 빌드가 멈춘다. 단계마다 '[finalize <phase>]'로 기록한다.
#
#   root 부분(USER 전환 전)
#     1) nmfc manifest --verify docker/context.lock --strict  (불일치 1건이라도 있으면 빌드 중단)
#        lock이 없으면 폴백: 빌드 중 생성(nmfc.manifest.source=in-build, 외부 대조 없는 약한 모드)
#     2) nmfc manifest --baseline /opt/nmfc/baseline, 끝나면 chmod -R a-w (root 소유 읽기 전용)
#     3) 자기검증 1부: uv pip check, freeze·torch.__config__.show()·labels.json 등을 /opt/nmfc/env/에 기록
#   USER nmfc 부분
#     4) matplotlib 폰트 캐시(~/.cache/matplotlib) 생성, 한글 폰트 단언
#        (regen이 저장하는 results/openset_noise_grid/*.png와, 체크포인트가 있을 때의
#        results/openset_detection/histogram_*.png에 한글 라벨이 있다)
#     5) 자기검증 2부: 타깃별 torch 상수 단언, uid 1000 쓰기 시험
#     6) 소스 트리 무변경 재검증(manifest --verify --strict), /out이 비어 있는지 확인
#
# 빌드 인자(ARG → RUN 환경변수): NMFC_MANIFEST_SHA256, NMFC_MANIFEST_SOURCE(host-export|in-build), SNAPSHOT,
#   NMFC_TITLE, NMFC_DESCRIPTION, NMFC_REVISION, NMFC_CREATED, NMFC_LICENSES

set -Eeuo pipefail
IFS=$'\n\t'

readonly REPO=${NMFC_FINALIZE_REPO:-/workspace/nmfc}
readonly PY=${NMFC_FINALIZE_PY:-/opt/venv/bin/python}
readonly ENV_DIR=/opt/nmfc/env
readonly BASELINE_DIR=/opt/nmfc/baseline
readonly EXPECTED_REL=docker/expected.yaml
readonly LOCK_REL=docker/context.lock
readonly FILES_REL=docker/manifest.files

PHASE=${PHASE:-?}
TARGET=${TARGET:-?}
CLEANUP_DIR=""

log() { printf '[finalize %s/%s] %s\n' "$TARGET" "$PHASE" "$*"; }
die() { printf '[finalize %s/%s] 실패: %s\n' "$TARGET" "$PHASE" "$*" >&2; exit 1; }
usage() { printf 'usage: bash docker/finalize.sh <cpu|gpu> <root|user>\n' >&2; exit 2; }

cleanup() {
  if [[ -n ${CLEANUP_DIR:-} && -d $CLEANUP_DIR ]]; then
    rm -rf -- "$CLEANUP_DIR"
  fi
}

sha256_of() {
  local out
  out=$(sha256sum -- "$1")
  printf '%s\n' "${out%% *}"
}

require_ids() {  # require_ids <uid> <gid>
  [[ $(id -u) == "$1" && $(id -g) == "$2" ]] \
    || die "이 단계는 uid:gid $1:$2에서 실행해야 한다(현재 $(id -u):$(id -g))"
}

assert_out_empty() {
  if [[ -d /out && -n $(ls -A /out) ]]; then
    die "/out에 빌드 중 산출물이 남았다: $(ls -A /out | tr '\n' ' ')"
  fi
}

# ------------------------------------------------------------------------------------------- 검사 함수(단위 시험 가능)

# manifest_consistency <lock> <source> <label_sha> — lock 파일과 빌드 인자가 서로 맞는지 본다.
#   host-export: lock이 있어야 하고 sha256(lock 바이트) == label_sha(64자리 소문자 16진수)
#   in-build   : label_sha는 비어 있어야 한다(라벨에 대조되지 않은 해시를 달지 않는다)
manifest_consistency() {
  local lock=$1 src=$2 want=$3 got
  case $src in
    host-export)
      [[ -f $lock ]] || die "host-export인데 $lock이 없다 — docker/build.ps1·build.sh로 빌드했는지 확인"
      [[ $want =~ ^[0-9a-f]{64}$ ]] || die "NMFC_MANIFEST_SHA256 형식이 잘못됐다: '$want'"
      got=$(sha256_of "$lock")
      [[ $got == "$want" ]] || die "context.lock sha256 불일치: 빌드 인자 $want, 컨텍스트 $got"
      ;;
    in-build)
      [[ -z $want ]] || die "in-build 모드에서는 NMFC_MANIFEST_SHA256을 비워 둬야 한다(받은 값: $want)"
      ;;
    *)
      die "NMFC_MANIFEST_SOURCE는 host-export 또는 in-build여야 한다: '$src'"
      ;;
  esac
}

# font_assert <expected.yaml> — verify.font_assert.families를 순서대로
#   findfont(fallback_to_default=False)로 찾고, 처음 찾은(=matplotlib가 실제로 쓰는) 폰트 파일에
#   verify.font_assert.char 글리프가 있어야 한다(FT2Font.get_char_index > 0). 이 호출이 폰트 캐시도 만든다.
font_assert() {
  "$PY" - "$1" <<'PY'
import sys
import yaml
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm
from matplotlib.ft2font import FT2Font
with open(sys.argv[1], encoding="utf-8") as f:
    fa = yaml.safe_load(f)["verify"]["font_assert"]
families, ch = fa["families"], fa["char"]
found = []
for fam in families:
    try:
        path = fm.findfont(fm.FontProperties(family=fam), fallback_to_default=False)
    except ValueError:
        print(f"  {fam}: 없음")
        continue
    idx = FT2Font(path).get_char_index(ord(ch))
    print(f"  {fam}: {path} (글리프 {ch!r} index={idx})")
    found.append((fam, path, idx))
if not found:
    sys.exit(f"한글 폰트 단언 실패: {families} 중 찾은 폰트가 없다(두부 글자가 생긴다)")
fam, path, idx = found[0]
if idx <= 0:
    sys.exit(f"한글 폰트 단언 실패: 1순위 폰트 {fam}({path})에 {ch!r} 글리프가 없다")
print(f"  한글 폰트 단언 통과: {fam}")
print(f"  matplotlib 캐시 디렉터리: {matplotlib.get_cachedir()}")
PY
}

# torch_assert <cpu|gpu> — 타깃별 상수 단언(빌드 중에는 GPU가 없어도 된다)
torch_assert() {
  "$PY" - "$1" <<'PY'
import sys
import torch
import torchvision
target = sys.argv[1]
if target == "gpu":
    want = {"torch": "2.11.0+cu128", "torchvision": "0.26.0+cu128", "cuda": "12.8", "cudnn": 91900}
    got = {"torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
           "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
elif target == "cpu":
    want = {"torch": "2.11.0+cpu", "torchvision": "0.26.0+cpu", "cuda": None}
    got = {"torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
           "cuda": torch.version.cuda}
else:
    sys.exit(f"알 수 없는 타깃: {target}")
bad = {k: (want[k], got[k]) for k in want if got[k] != want[k]}
print(f"  {target}: {got}")
if bad:
    sys.exit(f"타깃 상수 단언 실패({target}): " + ", ".join(f"{k} 기대 {w!r} / 실제 {g!r}" for k, (w, g) in bad.items()))
print("  타깃 상수 단언 통과")
PY
}

# write_test — uid 1000 쓰기 시험: 디렉터리가 있고, 쓰기 가능하며, 실제로 파일을 만들고 지울 수 있어야 한다
write_test() {
  local d f
  for d in results logs runs data/raw checkpoints \
           /cache/torch /cache/hf /out "$HOME/.cache"; do
    [[ $d == /* ]] || d="$REPO/$d"
    [[ -d $d ]] || die "쓰기 시험: $d 디렉터리가 없다"
    [[ -w $d ]] || die "쓰기 시험: uid $(id -u)가 $d에 쓸 수 없다"
    f=$(mktemp -p "$d" .nmfc-write-test.XXXXXX) || die "쓰기 시험: $d에 파일을 만들 수 없다"
    rm -f -- "$f"
  done
  log "  uid $(id -u) 쓰기 시험 통과"
}

# ------------------------------------------------------------------------------------------------ root 부분

root_phase() {
  require_ids 0 0
  cd "$REPO"
  CLEANUP_DIR=$(mktemp -d /tmp/nmfc-finalize-root.XXXXXX)
  trap cleanup EXIT
  export NMFC_OUT="$CLEANUP_DIR/out"   # nmfc 하위 명령이 이미지의 /out에 보고서를 남기지 않게 한다

  log "0) /usr/local/bin/nmfc 래퍼와 CLI 확인"
  [[ -x /usr/local/bin/nmfc ]] || die "/usr/local/bin/nmfc가 없다"
  nmfc --help > /dev/null

  log "1) 매니페스트"
  local src=${NMFC_MANIFEST_SOURCE:-in-build} want=${NMFC_MANIFEST_SHA256:-}
  manifest_consistency "$LOCK_REL" "$src" "$want"
  if [[ -f $LOCK_REL ]]; then
    [[ $src == host-export ]] || log "  lock이 컨텍스트에 있다 — 그대로 엄격 검증한다(라벨은 in-build)"
    nmfc manifest --verify "$LOCK_REL" --root "$REPO" --strict \
      || die "컨텍스트가 context.lock과 다르다 — 빌드를 중단한다"
  else
    log "  lock 없음 — 폴백: 빌드 중 생성(nmfc.manifest.source=in-build, 외부 대조 없는 약한 모드)"
    nmfc manifest --write-lock "$REPO"
  fi
  [[ -s $LOCK_REL && -s $FILES_REL ]] || die "$LOCK_REL 또는 $FILES_REL이 없다"
  chown 0:0 "$LOCK_REL" "$FILES_REL"
  chmod 0444 "$LOCK_REL" "$FILES_REL"
  MANIFEST_SHA=$(sha256_of "$LOCK_REL")
  # 실행 도구가 보고서 첫 줄에 적을 값(manifest_sha256)이 lock 해시와 같아야 한다.
  local tool_sha
  tool_sha=$(env -u NMFC_MANIFEST_SHA256 "$PY" -c \
    'import sys; sys.path.insert(0, "docker"); import nmfc_common as c; print(c.manifest_sha256() or "")')
  [[ $tool_sha == "$MANIFEST_SHA" ]] \
    || die "nmfc_common.manifest_sha256()=$tool_sha 이 sha256(context.lock)=$MANIFEST_SHA 과 다르다"
  log "  manifest sha256 $MANIFEST_SHA (source=$src, 파일 $(wc -l < "$FILES_REL")개)"

  log "2) 기준본 $BASELINE_DIR"
  rm -rf -- "$BASELINE_DIR"
  install -d -m 0755 "$(dirname -- "$BASELINE_DIR")"
  nmfc manifest --baseline "$BASELINE_DIR" --root "$REPO"
  local f
  for f in results/theory_audit.json results/retrieval_eval.json results/stats_ci.json results/seed_robustness.json \
           results/openset_noise_grid/master.json results/openset_noise_grid/master.csv \
           logs/c100v_shrb/history.json "$LOCK_REL"; do
    [[ -f $BASELINE_DIR/$f ]] || die "기준본에 $f가 없다"
  done
  chown -R 0:0 "$BASELINE_DIR"
  chmod -R a-w "$BASELINE_DIR"
  log "  기준본 파일 $(find "$BASELINE_DIR" -type f | wc -l)개(root 소유 읽기 전용)"

  log "3) 자기검증 1부 → $ENV_DIR"
  command -v uv > /dev/null || die "uv가 없다 — Dockerfile의 RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv 확인"
  rm -rf -- "$ENV_DIR"
  install -d -m 0755 "$ENV_DIR"
  if ! UV_NO_CACHE=1 uv pip check --python "$PY" > "$ENV_DIR/uv-pip-check.txt" 2>&1; then
    cat "$ENV_DIR/uv-pip-check.txt" >&2
    die "uv pip check 실패"
  fi
  UV_NO_CACHE=1 uv pip freeze --python "$PY" > "$ENV_DIR/freeze.txt"
  [[ -s $ENV_DIR/freeze.txt ]] || die "freeze가 비어 있다"
  "$PY" -c 'import torch; print(torch.__config__.show())' > "$ENV_DIR/torch-config.txt"
  "$PY" - "$TARGET" > "$ENV_DIR/versions.json" <<'PY'
import json
import platform
import sys
import matplotlib
import numpy
import torch
import torchvision
info = {
    "target": sys.argv[1],
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "libc": "-".join(platform.libc_ver()),
    "torch": str(torch.__version__),
    "torchvision": str(torchvision.__version__),
    "torch_cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    "numpy": numpy.__version__,
    "matplotlib": matplotlib.__version__,
}
print(json.dumps(info, ensure_ascii=False, indent=1))
PY
  ldd --version 2>&1 | sed -n '1p' > "$ENV_DIR/glibc.txt"
  dpkg-query -W -f='${Package}\t${Version}\n' > "$ENV_DIR/dpkg.txt"
  "$PY" - "$BACKEND" "$src" "$want" "$MANIFEST_SHA" "$(wc -l < "$FILES_REL")" "$ENV_DIR" <<'PY'
import json
import os
import sys
backend, src, label_sha, lock_sha, n_files, env_dir = sys.argv[1:7]
labels = {  # Dockerfile LABEL과 같은 값(컨테이너 안에서는 이미지 라벨을 읽을 수 없으므로 여기 남긴다)
    "org.opencontainers.image.title": os.environ.get("NMFC_TITLE", ""),
    "org.opencontainers.image.description": os.environ.get("NMFC_DESCRIPTION", ""),
    "org.opencontainers.image.revision": os.environ.get("NMFC_REVISION", ""),
    "org.opencontainers.image.created": os.environ.get("NMFC_CREATED", ""),
    "org.opencontainers.image.licenses": os.environ.get("NMFC_LICENSES", ""),
    "nmfc.manifest.sha256": label_sha,
    "nmfc.manifest.source": src,
    "nmfc.apt.snapshot": os.environ.get("SNAPSHOT", ""),
    "nmfc.backend": backend,
}
manifest = {"source": src, "label_sha256": label_sha, "lock_sha256": lock_sha, "files": int(n_files)}
for name, obj in (("labels.json", labels), ("manifest.json", manifest)):
    with open(os.path.join(env_dir, name), "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.write("\n")
PY
  chown -R 0:0 "$ENV_DIR"
  chmod -R a-w "$ENV_DIR"
  log "  $(sed -n '1p' "$ENV_DIR/glibc.txt"); freeze $(wc -l < "$ENV_DIR/freeze.txt")줄"

  assert_out_empty
  log "root 부분 완료"
}

# ------------------------------------------------------------------------------------------------ USER nmfc 부분

user_phase() {
  require_ids 1000 1000
  if [[ ${HOME:-} != /home/nmfc ]]; then
    log "HOME=${HOME:-(없음)} → /home/nmfc"
    export HOME=/home/nmfc
  fi
  cd "$REPO"
  CLEANUP_DIR=$(mktemp -d /tmp/nmfc-finalize-user.XXXXXX)
  trap cleanup EXIT
  local tmp=$CLEANUP_DIR
  export NMFC_OUT="$tmp/out"

  log "4) matplotlib 폰트 캐시와 한글 폰트 단언"
  font_assert "$EXPECTED_REL"
  compgen -G "$HOME/.cache/matplotlib/fontlist-*.json" > /dev/null \
    || die "matplotlib 폰트 캐시가 $HOME/.cache/matplotlib에 없다"

  log "5) 자기검증 2부: 타깃 상수와 uid 1000 쓰기 시험"
  torch_assert "$TARGET"
  write_test

  log "6) 빌드가 소스 트리를 바꾸지 않았는지 재검증"
  nmfc manifest --verify "$LOCK_REL" --root "$REPO" --strict \
    || die "finalize 중에 소스 트리가 바뀌었다"
  assert_out_empty
  log "USER nmfc 부분 완료"
}

main() {
  [[ $# -eq 2 ]] || usage
  TARGET=$1
  PHASE=$2
  case $TARGET in
    cpu) BACKEND=cpu ;;
    gpu) BACKEND=cu128 ;;
    *) usage ;;
  esac
  trap 'rc=$?; printf "[finalize %s/%s] 오류: %s 줄 %s: %s (exit %s)\n" "$TARGET" "$PHASE" "${BASH_SOURCE[0]}" "$LINENO" "$BASH_COMMAND" "$rc" >&2' ERR
  case $PHASE in
    root) root_phase ;;
    user) user_phase ;;
    *) usage ;;
  esac
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  main "$@"
fi
