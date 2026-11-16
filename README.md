# NMFC 실험 재현 저장소

NMFC(Nonlinear Marginal Fisher Classification) 논문 실험의 실행 코드·설정, 실험 원자료(`results/`·`logs/`),
Docker 재현 도구를 담고 있습니다. 명령을 실행하면 저장소에 커밋된 결과(`results/`)와 새로 계산한 값을 나란히 적은
결과표가 만들어집니다. 설명·해석 문서는 담지 않았습니다.

## 준비물

- x86-64(인텔·AMD) PC와 Docker (Windows는 Docker Desktop + WSL2)
- GPU 명령(`regen`·`train`·`reproduce`·`smoke`): NVIDIA GPU(Turing 이상, VRAM 16 GB 기준), CUDA 12.8을 지원하는 드라이버.
  Linux는 NVIDIA Container Toolkit도 필요합니다.
- `verify`·`regen-cpu`는 CPU만으로, 네트워크 없이 실행됩니다.
- 이미지를 직접 만들 때: git, Python 3(표준 라이브러리만 사용)

## 저장소 받기

```bash
git clone https://github.com/HwaminCLaude/NMFC.git
cd NMFC
```

아래 명령은 모두 저장소 루트에서 실행합니다(`compose.yaml`이 여기에 있습니다).

## 이미지 받기

[Releases](https://github.com/HwaminCLaude/NMFC/releases)에서 이미지 파일과 `SHA256SUMS`를 받습니다.

- `nmfc-cpu.tar.gz` → `nmfc:cpu`
- `nmfc-gpu.tar.gz.part-00`·`part-01`·`part-02` → 합쳐서 `nmfc-gpu.tar.gz` → `nmfc:gpu`

Git Bash·Linux:

```bash
sha256sum -c --ignore-missing SHA256SUMS
cat nmfc-gpu.tar.gz.part-* > nmfc-gpu.tar.gz
sha256sum -c --ignore-missing SHA256SUMS        # 합친 파일도 확인
docker load -i nmfc-cpu.tar.gz
docker load -i nmfc-gpu.tar.gz
```

Windows PowerShell:

```powershell
cmd /c copy /b nmfc-gpu.tar.gz.part-00+nmfc-gpu.tar.gz.part-01+nmfc-gpu.tar.gz.part-02 nmfc-gpu.tar.gz
Get-FileHash nmfc-gpu.tar.gz -Algorithm SHA256   # SHA256SUMS의 nmfc-gpu.tar.gz 값과 같아야 합니다
docker load -i nmfc-cpu.tar.gz
docker load -i nmfc-gpu.tar.gz
```

## 이미지 직접 만들기

```powershell
powershell -ExecutionPolicy Bypass -File docker\build.ps1 -Target cpu   # nmfc:cpu
powershell -ExecutionPolicy Bypass -File docker\build.ps1 -Target gpu   # nmfc:gpu
```

Linux에서는 `bash docker/build.sh cpu`, `bash docker/build.sh gpu`를 씁니다. 빌드는 git 인덱스의 파일만 씁니다.

## 실행

Linux에서는 먼저 `mkdir -p out && chmod a+rwx out`을 실행합니다(컨테이너는 uid 1000으로 `out/`에 씁니다).

```bash
docker compose run --rm verify
docker compose run --rm regen
docker compose run --rm fetch data
docker compose run --rm train --stage T1 --force
docker compose run --rm compare
```

| 서비스 | 하는 일 | 이미지 | 결과표 (`out/` 아래) |
|---|---|---|---|
| `verify` | 단위 테스트(`tests/`)와 명제 검증 스크립트를 실행하고 `results/theory_audit.json`과 대조 | cpu | `verify/theory_claims` |
| `regen-cpu`, `regen` | 저장소의 원자료(임베딩·`metrics.json`)로 집계 평가 스크립트를 다시 실행하고 커밋된 집계 파일과 대조 | cpu / gpu | `regen/aggregates`, `regen/table_files` |
| `fetch` | 데이터(`data`)·ImageNet 가중치(`weights`) 내려받기와 sha256 검증 | gpu | — |
| `smoke`, `env` | GPU·데이터 로더·환경 점검 | gpu | — |
| `train` | 학습 단계(T1~T8) 실행 | gpu | `train/trained_runs` |
| `compare` | `train`·`reproduce` 결과를 커밋된 `results/`와 대조 | gpu | `compare/compare_table`, `compare/gates` |
| `reproduce` | 학습 T1~T8 + 집계·감사 S1·S3·S4 (외부 임베딩 S2는 `--with-ext`를 붙일 때만) | gpu | (`compare`로 확인) |
| `shell` | 이미지 안 bash | gpu | — |

- `fetch data`는 약 400 MB(CIFAR-10·CIFAR-100·SVHN 테스트)를 받습니다. 받는 속도는 데이터 서버에 따라 다릅니다.
  같은 압축본이 이미 있으면 그 폴더를 `NMFC_HOST_DATA` 환경변수로 지정하고 `docker compose run --rm fetch data --from /host-data`로 복사합니다.
- CIFAR 학습(T2~T8)에는 먼저 `fetch data`와 `fetch weights`가 필요합니다. T1(합성 데이터)은 필요 없습니다.

학습 단계(`train --stage`)와 런 수입니다. 전체 학습은 RTX 5080 기준 1~2일이 걸립니다(`scripts/run_all.ps1` 주석).

| 단계 | 내용 | 런 |
|---|---|---|
| T1 | 합성 2D 데이터(moons·circles·swiss) | 9 |
| T2 | CIFAR-100 v1 기본 설정과 기본 lr(1e-3) 기준선 3종 | 4 |
| T3 | CIFAR-100 교차 엔트로피(lr 3e-4) | 1 |
| T4 | CIFAR-100 DML 기준선(lr 3e-4) | 4 |
| T5 | 변형·from-scratch·OOD·개방셋 | 15 |
| T6 | 시드 복제(시드 0~3) | 24 |
| T7 | 라벨 오염 그리드(시드 42) | 16 |
| T8 | 개방셋·라벨 오염 시드 복제(시드 0~2) | 54 |

`train`·`reproduce`의 결과는 이름 있는 볼륨에 남습니다. 새로 시작할 때는 `compose.yaml` 머리 주석의
`docker volume rm` 명령으로 볼륨을 지웁니다.

## 결과표 읽기

- 위치: `out/<명령>/<표>.csv`(엑셀에서 바로 열림)와 같은 내용의 `.md`. `.md` 파일의 둘째 줄은 판정별 건수입니다.
- 실행 보고서: `out/<명령>_report.md`·`.json` (`fetch`는 `fetch_data_report`·`fetch_weights_report`.
  `regen`과 `regen-cpu`는 같은 이름에 쓰므로 나중 실행이 앞의 것을 덮어씁니다)
- 열: 표마다 앞쪽에 식별 열(파일·경로·실험·지표 등)이 있고, 이어서 `커밋 값(results/)`, `새 값`, `차이`(새 값 − 커밋 값),
  `허용범위`, `판정`, `비고`가 옵니다. `train/trained_runs`에는 `허용범위`가 없고(train은 판정하지 않음),
  `compare/gates`는 `게이트`, `통계량`, `값`, `허용범위`, `판정`, `비고`입니다.
- `판정` 값
  - `PASS`: 허용범위 안
  - `FAIL`: 허용범위 밖
  - `REPORT`: 판정 없이 값만 보고
  - `N/A`: 평가하지 않음(예: 체크포인트가 없어 실행되지 않은 단계)
- `regen`의 실행 라벨: `PASS_EXACT`(커밋 값과 모든 값이 같음), `PASS_TOLERANCE`(차이가 허용범위 안), `FAIL`
- 종료 코드: 0 = PASS, 1 = FAIL, 2 = 판정 불가(전제조건 불충족, `compare`에서 평가하지 못한 게이트), 130 = 사용자 중단
- 허용범위의 정의: `docker/expected.yaml`(verify), `docker/tolerances.yaml`(regen·compare)

## 폴더

| 폴더 | 내용 |
|---|---|
| `src/` | 모델·손실·학습·평가 코드 |
| `configs/` | 실험 설정 |
| `scripts/` | 평가·감사 스크립트, 전체 실행 순서(`run_all.ps1`) |
| `tests/` | 단위 테스트 |
| `results/` | 실험별 `metrics.json`(일부 `embeddings.npz`)과 집계 파일 |
| `logs/` | 실험별 학습 기록 `history.json` |
| `docker/` | 재현 도구(`nmfc` 명령), 허용범위 정의, 도구 테스트 |
| `requirements/` | 이미지용 고정 의존성(해시 포함) |
| `.github/` | CI(cpu 이미지 빌드와 verify) |

## 실험 이름 규칙

변형 코드 4글자는 `[energy, logit, apt, bank]` 순서입니다(`src/variants.py`).

| 자리 | 글자 |
|---|---|
| energy | `m` = mean, `s` = soft-min |
| logit | `m` = mean_scaled, `h` = hinge·temperature |
| apt(1단계 워밍업 손실) | `r` = reciprocal, `j` = −J (코드에는 `n`·`t`도 있지만 이 저장소의 런에는 없습니다) |
| bank(학습 중 참조) | `b` = 미니배치, `o` = 온라인 뱅크 |

예: `c100v_shrb` = 제안 방법 설정, `c100v_mhrb` = energy만 mean으로 바꾼 설정. `cifar100_nmfc`는 v1 기본 설정입니다.

| 접두사 | 뜻 |
|---|---|
| `c100v_` | CIFAR-100 변형(4글자 코드) 실험과 개방셋 실험 (ImageNet 사전학습 ResNet-50, 224px) |
| `c100s_` | CIFAR-100, 사전학습 없이(200 에폭) |
| `ood10v_` | CIFAR-10 학습, SVHN을 분포 밖 데이터로 사용 |
| `cifar100_` | CIFAR-100 v1 계열: v1 NMFC(`cifar100_nmfc`), 기본 lr 1e-3 기준선(`triplet`·`proto`·`deepmfa`), lr 3e-4로 맞춘 기준선(`_lr3`·`_plr`) |
| `ood_cifar10_` | OOD 실험의 softmax 대조군 |
| `syn_` | 합성 2D 데이터 |
| `diag224_` | 224px 5에폭 진단 실행(학습 큐 밖) |
| `ext_` | 외부 사전학습 모델 임베딩(재학습 없음) |

| 접미사 | 뜻 |
|---|---|
| `_s0`~`_s3` | 학습 시드(없으면 42) |
| `_open` | 개방셋: 80개 클래스만 학습, 100개로 평가 |
| `_ns5`~`_ns20` / `_na5`~`_na20` | 라벨 오염 비율(%) — 학습 클래스만 / 전체 클래스 |
| `_bank` | 개방셋 체크포인트에 참조 뱅크 추론만 적용(재학습 없음) |
| `_lr3` | 학습률 3e-4 |
| `_plr` | Proxy-Anchor 프록시 학습률 ×100 |

방법 이름: `nmfc`(변형 코드 없는 v1 NMFC), `softmax`(교차 엔트로피), `linear`(은닉층 없는 softmax, 합성 데이터용),
`arcface`, `normsoftmax`, `ms`(Multi-Similarity), `proxyanchor`, `triplet`, `proto`(Prototypical), `deepmfa`.

## Docker 없이 실행 (Windows)

[uv](https://docs.astral.sh/uv/)와 NVIDIA GPU가 필요합니다. `setup_env.ps1`은 `.venv`를 만들고 CUDA 12.8용 torch와
`requirements.txt`를 설치한 뒤 GPU 점검을 합니다. 실측 환경의 정확한 버전은 `requirements-lock.txt`에 있습니다.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe docker\nmfc_cli.py verify --out out
.venv\Scripts\python.exe docker\nmfc_cli.py regen --device cuda --out out
.venv\Scripts\python.exe docker\nmfc_cli.py train --stage T1 --force --keep --out out
.venv\Scripts\python.exe docker\nmfc_cli.py compare --workdir "<train이 출력한 작업 사본 경로>" --out out
```

호스트에서는 명령마다 저장소를 임시 폴더에 복사해 그 사본에서 실행합니다. `train`에 `--keep`을 붙이면 사본이 남고,
`train`이 마지막에 출력하는 `compare --workdir …` 명령으로 그 사본을 대조합니다. 전체 학습·평가 순서는 `scripts\run_all.ps1`에 있습니다.

## 인용·라이선스

인용 정보는 `CITATION.cff`, 라이선스는 MIT(`LICENSE`)입니다.
