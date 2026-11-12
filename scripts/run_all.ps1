# NMFC 실험 전체 재현 파이프라인 — 학습(캐시 스킵) → 뱅크 평가 → 외부 임베딩 → 집계 평가 → 감사·명제 검증
#
# results/<exp>/metrics.json 캐시가 있으면 해당 학습은 자동으로 건너뜁니다.
# 처음부터 전부 재학습하려면 해당 실험의 results/<exp>/·checkpoints/<exp>/·logs/<exp>/ 를
# 지우고 실행하세요 (이 저장소는 metrics.json 을 포함하고 있으므로 그냥 돌리면 전부 스킵됩니다).
#
# 총 127개 학습 런. RTX 5080 기준 대략 1~2일 (224px·40에폭 1런당 30~40분,
# from-scratch 200에폭 3런과 라벨오염 그리드 72런이 대부분을 차지합니다).
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"

# ---- 1) 본 실험 ------------------------------------------------------------
# 세 config는 저장소에 담지 않은 런도 함께 정의한다 — config 자체는
# 역사적 기록이라 손대지 않고, 여기서 --only 로 재현 대상 런만 지명한다.
# 지명하지 않은 런: syn_moons_apt_{none,fixed3} / cifar100_{softmax,arcface} / cifar100_proxyanchor_lr3.
# 합성 2D 데이터 (9런)
& $py -m src.train --config configs\synthetic.yaml --only `
    syn_moons_linear syn_circles_linear syn_swiss_linear `
    syn_moons_softmax syn_circles_softmax syn_swiss_softmax `
    syn_moons_nmfc syn_circles_nmfc syn_swiss_nmfc
# v1 기본 설정 + 기본 lr(1e-3) 기준선 3종 (4런)
& $py -m src.train --config configs\cifar100.yaml --only `
    cifar100_nmfc cifar100_triplet cifar100_proto cifar100_deepmfa
# CE(lr 정렬)
& $py -m src.train --config configs\cifar100_softmax_lr3.yaml
# DML 4런 (ArcFace·NormSoftmax·MS·Proxy-Anchor plr)
& $py -m src.train --config configs\cifar100_dml_lr3.yaml --only `
    cifar100_arcface_lr3 cifar100_normsoftmax_lr3 cifar100_ms_lr3 cifar100_proxyanchor_plr
& $py -m src.train --config configs\cifar100_variants.yaml            # 제안 shrb·절제 mhrb + negJ 변형 shjb·shjo
& $py -m src.train --config configs\cifar100_variants_wave2.yaml      # 변형 shro (온라인 뱅크 단독 격리)
& $py -m src.train --config configs\cifar100_scratch.yaml             # from-scratch 2런 (200에폭)
& $py -m src.train --config configs\cifar100_dml_scratch.yaml --only c100s_proxyanchor  # from-scratch Proxy-Anchor
& $py -m src.train --config configs\ood_cifar10_softmax_lr3.yaml      # OOD MSP(lr 정렬)
& $py -m src.train --config configs\ood_cifar10_variants.yaml         # OOD shrb·mhrb + negJ 변형 shjb·shjo
& $py -m src.train --config configs\cifar100_openset.yaml             # 개방셋 2런

# ---- 2) 시드 복제 (5시드 — 시드 42는 위에서 이미 학습됨) --------------------
foreach ($s in 0, 1, 2, 3) {
    & $py -m src.train --config "configs\cifar100_seeds_s$s.yaml"        # shrb·mhrb·CE
    & $py -m src.train --config "configs\ood_cifar10_seeds_s$s.yaml"     # OOD shrb·mhrb·CE
}

# ---- 3) 라벨오염 그리드 (72런, 가장 오래 걸립니다) -------------------------
& $py -m src.train --config configs\cifar100_openset_noise.yaml          # ns5·na5 (4런)
& $py -m src.train --config configs\cifar100_openset_noise_grid.yaml     # ns/na × {10,15,20} (12런)
foreach ($s in 0, 1, 2) {
    & $py -m src.train --config "configs\cifar100_openset_seeds_s$s.yaml"        # clean·ns5·na5·ns20·na20 시드
    & $py -m src.train --config "configs\cifar100_openset_noise_grid_s$s.yaml"   # ns/na × {10,15} 시드
}

# ---- 4) 개방셋 참조뱅크 이식 평가 (위 openset 체크포인트 필요) --------------
& $py scripts\eval_softmax_bank.py                     # -> results/c100v_softmax_open_bank
& $py scripts\eval_softmax_bank.py c100v_shrb_open     # -> results/c100v_shrb_open_bank

# ---- 5) 외부 사전학습 임베딩 (동결 인코더 비교용, 인터넷 필요) --------------
& $py scripts\embed_external.py                        # -> results/ext_*/embeddings.npz

# ---- 6) 임베딩 집계 평가 (results/<exp>/embeddings.npz 기반, 재학습 0) ------
& $py scripts\eval_retrieval.py            # -> results/retrieval_eval.json       (R@1/mAP/실루엣)
& $py scripts\eval_fewshot_verify.py       # -> results/fewshot_verify_eval.json  (few-shot·검증)
& $py scripts\eval_stats_ci.py             # -> results/stats_ci.json             (20에피소드·부트스트랩 CI)
& $py scripts\eval_seed_robustness.py      # -> results/seed_robustness.json      (5시드 평균·SD)
& $py scripts\eval_near_ood.py             # -> results/near_ood.json             (near-OOD AUROC)
& $py scripts\eval_calibration.py          # -> results/calibration_eval.json     (오분류 탐지)
& $py scripts\eval_openset_detection.py    # -> results/openset_detection/        (개방셋 탐지 AUROC)
& $py scripts\eval_openset_noise_grid.py   # -> results/openset_noise_grid/       (라벨오염 그리드)

# ---- 7) 감사 + 명제 검증 ---------------------------------------------------
& $py scripts\audit_operating_regime.py       # -> results/operating_regime_audit.json (작동점)
& $py scripts\audit_theory_claims.py          # -> results/theory_audit.json  (명제 16/16)
