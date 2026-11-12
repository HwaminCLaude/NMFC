# NMFC 실험 재현 이미지
#
# 타깃은 두 개다(백엔드는 스테이지 이름에 하드코딩한다. FROM의 ARG로는 타깃별 백엔드를 고를 수 없다).
#   cpu : deps-cpu   (torch 2.11.0+cpu)    verify · regen --device cpu
#   gpu : deps-cu128 (torch 2.11.0+cu128)  env · regen --device cuda · smoke · fetch · train · reproduce · compare
#
# 빌드 경로는 하나다: docker/build.ps1(Windows) 또는 docker/build.sh(Linux·CI).
#   git 인덱스 → ASCII 경로로 내보내기 → docker/context.lock 생성 →
#   docker build --provenance=false --target <cpu|gpu>
#     --build-arg NMFC_MANIFEST_SHA256=<sha256(context.lock)> --build-arg NMFC_MANIFEST_SOURCE=host-export
#     --build-arg NMFC_REVISION=<git rev-parse --short=12 HEAD>[+index] | no-commit
#   (+index: 내보낸 인덱스가 HEAD와 다름. no-commit: HEAD가 없음)
# 클론에서 `docker build --target cpu .`를 직접 하면 finalize가 빌드 중에 lock을 만들고
# 라벨 nmfc.manifest.source=in-build를 단다(외부 대조가 없는 약한 모드). 이때 리비전 라벨은 기본값 unknown이다.
#
# 외부 입력은 모두 고정한다.
#   - 베이스·uv 이미지: digest
#   - apt: snapshot.debian.org 고정 시점(SNAPSHOT) + 데비안 아카이브 키 서명 검증
#   - 파이썬 휠: requirements/linux-*.lock.txt의 --hash(uv pip sync --require-hashes)
# BuildKit이 필요하다(RUN --mount). `# syntax=` 지시어는 쓰지 않는다. 쓰면 고정되지 않은 프런트엔드 이미지를 받는다.

ARG SNAPSHOT=20261110T000000Z
ARG NMFC_TITLE=nmfc-repro
ARG NMFC_DESCRIPTION="Reproduction image for the NMFC experiments: nmfc env | verify | regen | fetch | smoke | train | reproduce | compare"
ARG NMFC_REVISION=unknown
ARG NMFC_CREATED=2026-08-31T04:39:35Z
ARG NMFC_LICENSES=MIT

# ---------------------------------------------------------------------------- uv (0.11.28 — 실측 venv를 만든 버전)
FROM astral/uv:0.11.28@sha256:0f36cb9361a3346885ca3677e3767016687b5a170c1a6b88465ec14aefec90aa AS uv

# ---------------------------------------------------------------------------- os: apt(snapshot)·사용자
FROM python:3.14.6-slim-bookworm@sha256:4c92ffcde4dd6f1ff72a24518f49fd4990b27134987dfa31a733badde66df9f8 AS os
ARG SNAPSHOT
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# apt 소스를 snapshot.debian.org 고정 시점으로 바꾼다(debian: bookworm·bookworm-updates, debian-security).
# 일반 미러는 스위트마다 최신 버전만 두므로 버전 고정이 시간이 지나면 깨진다.
RUN set -eux; \
    rm -f /etc/apt/sources.list /etc/apt/sources.list.d/*; \
    printf '%s\n' \
      'Types: deb' \
      "URIs: http://snapshot.debian.org/archive/debian/${SNAPSHOT}" \
      'Suites: bookworm bookworm-updates' \
      'Components: main' \
      'Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg' \
      'Check-Valid-Until: no' \
      '' \
      'Types: deb' \
      "URIs: http://snapshot.debian.org/archive/debian-security/${SNAPSHOT}" \
      'Suites: bookworm-security' \
      'Components: main' \
      'Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg' \
      'Check-Valid-Until: no' \
      > /etc/apt/sources.list.d/nmfc-snapshot.sources; \
    apt-get -o Acquire::Retries=5 update; \
    apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
      fonts-nanum fontconfig ca-certificates; \
    rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 nmfc \
 && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash nmfc

# ---------------------------------------------------------------------------- deps: 잠금 파일 그대로 venv 설치
# PyPI 기본 torch==2.11.0은 CUDA 13 빌드이므로 반드시 --torch-backend로 고른다.
# UV_COMPILE_BYTECODE=1: 실행 이미지는 PYTHONDONTWRITEBYTECODE=1이고 venv는 root 소유라 .pyc를 쓸 수 없다.
# 설치할 때 미리 컴파일해 두지 않으면 프로세스마다 torch 등을 소스에서 다시 컴파일한다.
FROM os AS deps-cpu
COPY --from=uv /uv /uvx /bin/
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_COMPILE_BYTECODE=1
COPY requirements/linux-cpu.lock.txt /tmp/requirements/linux-cpu.lock.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv --python /usr/local/bin/python3.14 \
 && uv pip sync --python /opt/venv/bin/python --require-hashes --torch-backend cpu \
      /tmp/requirements/linux-cpu.lock.txt

FROM os AS deps-cu128
COPY --from=uv /uv /uvx /bin/
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_COMPILE_BYTECODE=1
COPY requirements/linux-cu128.lock.txt /tmp/requirements/linux-cu128.lock.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/venv --python /usr/local/bin/python3.14 \
 && uv pip sync --python /opt/venv/bin/python --require-hashes --torch-backend cu128 \
      /tmp/requirements/linux-cu128.lock.txt

# ---------------------------------------------------------------------------- app: 저장소·디렉터리·래퍼(venv 없음)
FROM os AS app
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    TORCH_HOME=/cache/torch HF_HOME=/cache/hf \
    SOURCE_DATE_EPOCH=1788151175 \
    NMFC_IN_CONTAINER=1 NMFC_BASELINE=/opt/nmfc/baseline NMFC_OUT=/out \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
# verify·regen은 저장소 트리(results/ 아래 JSON·CSV·MD·PNG)에 다시 쓰므로 uid 1000 소유로 복사한다.
COPY --chown=1000:1000 . /workspace/nmfc
# 이미지에 없는 마운트 지점은 새 볼륨 루트가 root 소유가 되므로 미리 uid 1000으로 만든다.
RUN set -eux; \
    chown 1000:1000 /workspace/nmfc; \
    install -d -o 1000 -g 1000 \
      /workspace/nmfc/data /workspace/nmfc/data/raw /workspace/nmfc/checkpoints /workspace/nmfc/runs \
      /cache/torch /cache/hf /out; \
    printf '%s\n' '#!/bin/sh' 'exec /opt/venv/bin/python /workspace/nmfc/docker/nmfc_cli.py "$@"' \
      > /usr/local/bin/nmfc; \
    chmod 0755 /usr/local/bin/nmfc
WORKDIR /workspace/nmfc

# ---------------------------------------------------------------------------- cpu
FROM app AS cpu
ARG SNAPSHOT
ARG NMFC_TITLE
ARG NMFC_DESCRIPTION
ARG NMFC_REVISION
ARG NMFC_CREATED
ARG NMFC_LICENSES
ARG NMFC_MANIFEST_SHA256=""
ARG NMFC_MANIFEST_SOURCE=in-build
COPY --from=deps-cpu /opt/venv /opt/venv
# root 부분: manifest 검증 → 기준본 → 자기검증 1부(uv는 이미지에 남기지 않고 마운트만 한다)
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv bash /workspace/nmfc/docker/finalize.sh cpu root
USER nmfc
# USER nmfc 부분: matplotlib 폰트 캐시·한글 폰트 단언 → 자기검증 2부 → 소스 트리 무변경 재검증
RUN bash /workspace/nmfc/docker/finalize.sh cpu user
LABEL org.opencontainers.image.title="${NMFC_TITLE}" \
      org.opencontainers.image.description="${NMFC_DESCRIPTION}" \
      org.opencontainers.image.revision="${NMFC_REVISION}" \
      org.opencontainers.image.created="${NMFC_CREATED}" \
      org.opencontainers.image.licenses="${NMFC_LICENSES}" \
      nmfc.manifest.sha256="${NMFC_MANIFEST_SHA256}" \
      nmfc.manifest.source="${NMFC_MANIFEST_SOURCE}" \
      nmfc.apt.snapshot="${SNAPSHOT}" \
      nmfc.backend="cpu"
WORKDIR /workspace/nmfc
ENTRYPOINT ["nmfc"]
CMD ["verify"]

# ---------------------------------------------------------------------------- gpu
FROM app AS gpu
ARG SNAPSHOT
ARG NMFC_TITLE
ARG NMFC_DESCRIPTION
ARG NMFC_REVISION
ARG NMFC_CREATED
ARG NMFC_LICENSES
ARG NMFC_MANIFEST_SHA256=""
ARG NMFC_MANIFEST_SOURCE=in-build
COPY --from=deps-cu128 /opt/venv /opt/venv
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv bash /workspace/nmfc/docker/finalize.sh gpu root
USER nmfc
RUN bash /workspace/nmfc/docker/finalize.sh gpu user
LABEL org.opencontainers.image.title="${NMFC_TITLE}" \
      org.opencontainers.image.description="${NMFC_DESCRIPTION}" \
      org.opencontainers.image.revision="${NMFC_REVISION}" \
      org.opencontainers.image.created="${NMFC_CREATED}" \
      org.opencontainers.image.licenses="${NMFC_LICENSES}" \
      nmfc.manifest.sha256="${NMFC_MANIFEST_SHA256}" \
      nmfc.manifest.source="${NMFC_MANIFEST_SOURCE}" \
      nmfc.apt.snapshot="${SNAPSHOT}" \
      nmfc.backend="cu128"
WORKDIR /workspace/nmfc
ENTRYPOINT ["nmfc"]
CMD ["verify"]
