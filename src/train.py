"""학습 오케스트레이터 — 합성/이미지/OOD 러너와 2단계 NMFC 학습 루프.

실행:  python -m src.train --config configs/synthetic.yaml [--only NAME ...] [--force]
파이썬 코드에서는 run_from_config()를 호출한다 (results/ 캐시가 있으면 즉시 반환).
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Subset

from src.datasets.image import (
    IndexedDataset,
    get_image_datasets,
    get_plain_train_dataset,
    make_eval_loader,
    make_train_loader,
    resolve_label_noise,
    select_seen_classes,
)
from src.datasets.samplers import PKBatchSampler
from src.datasets.synthetic import get_synthetic
from src.evaluate import embed_loader, ood_auc, silhouette_of_embeddings, topk_accuracy
from src.models import (
    ArcFaceHead,
    EncoderMLP,
    LinearHead,
    NMFCHead,
    NormSoftmaxHead,
    OnlineClassBank,
    ProtoHead,
    ProxyAnchorHead,
    ResNet50Encoder,
    VariantNMFCHead,
    batch_hard_triplet_loss,
    mfa_regularizer,
    multi_similarity_loss,
    proxy_anchor_loss,
)
from src.nmfc import (
    APTController,
    FixedTransition,
    NoWarmup,
    fisher_j,
    fisher_ratio,
    linear_mfa_loss,
    neg_j_loss,
    nmfc_loss,
)
from src.nmfc.apt import LINEAR, NONLINEAR
from src.utils import PROJECT_ROOT, run_or_load, save_json, set_seed
from src.utils.experiments import exp_paths, load_embeddings
from src.utils.misc import get_device
from src.variants import (
    build_nmfc_head,
    build_warmup,
    is_negj_family,
    is_v1,
    resolve_variant,
    validate_variant,
    variant_code,
)

PK_METHODS = {"nmfc", "triplet", "deepmfa", "ms", "proxyanchor"}


# ---------------------------------------------------------------- config

def load_config(path: str | Path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "base" in cfg:
        base = yaml.safe_load((path.parent / cfg["base"]).read_text(encoding="utf-8"))
        cfg = {**base, **cfg}
    return cfg


def make_nmfc_head(spec: dict, cfg: dict) -> NMFCHead:
    nm = {**cfg["nmfc"], **spec.get("nmfc", {})}  # 실험별 오버라이드 허용
    return NMFCHead(lam=nm["lam"], n_ref=nm["n_ref"], detach_pi=nm["detach_pi"])


def make_apt(spec: dict, cfg: dict):
    a = {**cfg["apt"], **spec.get("apt", {})}  # 실험별 오버라이드 허용
    mode = spec.get("apt_mode", "auto")
    if mode == "fixed":
        return FixedTransition(fixed_epoch=spec["fixed_epoch"], window=a["window"])
    if mode == "none":
        return NoWarmup(window=a["window"])
    return APTController(
        window=a["window"],
        delta=a["delta"],
        patience=a["patience"],
        min_warmup=a["min_warmup"],
        max_warmup=a["max_warmup"],
    )


def _new_history() -> dict:
    return {
        "epoch": [],
        "loss": [],
        "lr": [],
        "phase": [],
        "fisher_ratio": [],
        "phi": [],
        "test_top1": [],
    }


# ---------------------------------------------------------------- 합성 러너

def run_synthetic(spec: dict, cfg: dict, paths: dict) -> dict:
    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = get_device()
    data = get_synthetic(spec["dataset"], n=spec["n_samples"], seed=seed)
    x_tr = torch.from_numpy(data["x_train"]).to(device)
    y_tr = torch.from_numpy(data["y_train"]).to(device)
    x_te = torch.from_numpy(data["x_test"]).to(device)
    y_te = torch.from_numpy(data["y_test"]).to(device)
    n_classes = int(y_tr.max().item()) + 1

    encoder = EncoderMLP(2, tuple(spec["hidden"]), spec["embed_dim"]).to(device)
    epochs = spec["epochs"]
    opt_cfg = {**cfg["optim"], **spec.get("optim", {})}  # 실험별 오버라이드 허용
    sampler = PKBatchSampler(data["y_train"], p=n_classes, k=spec["batch_k"], seed=seed)
    history = _new_history()
    method = spec["method"]

    if method == "softmax":
        head = LinearHead(spec["embed_dim"], n_classes).to(device)
        params = list(encoder.parameters()) + list(head.parameters())
        optimizer = torch.optim.AdamW(params, lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        apt = None
    else:  # nmfc
        head = make_nmfc_head(spec, cfg)
        optimizer = torch.optim.AdamW(
            encoder.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"]
        )
        apt = make_apt(spec, cfg)
        # 무워밍업(NoWarmup)은 처음부터 Phase II — 다른 군과 같은 cosine 예산을 받아야
        # 절제 비교가 lr 스케줄에 오염되지 않는다
        if apt.phase == NONLINEAR:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        else:
            scheduler = None  # Phase II 전환 시 cosine으로 생성

    for epoch in range(1, epochs + 1):
        encoder.train()
        phase = apt.phase if apt else "single"
        losses = []
        for idx in sampler:
            idx = torch.as_tensor(idx, device=device)
            z = encoder(x_tr[idx])
            yb = y_tr[idx]
            if method == "softmax":
                loss = F.cross_entropy(head(z), yb)
            elif phase == LINEAR:
                loss = linear_mfa_loss(z, yb)
            else:
                g, y_local = head.forward_batch(z, yb)
                loss = nmfc_loss(g, y_local)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), opt_cfg["grad_clip"])
            optimizer.step()
            losses.append(loss.item())

        # 에폭 말: Fisher ratio 프로브(전체 train) + APT 갱신 + 평가
        encoder.eval()
        transitioned = False
        with torch.no_grad():
            z_all = encoder(x_tr).float()
            r = fisher_ratio(z_all, y_tr).item()
            if apt:
                new_phase = apt.update(r)
                if new_phase == NONLINEAR and phase == LINEAR:
                    transitioned = True
                    optimizer = torch.optim.AdamW(
                        encoder.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"]
                    )
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=max(epochs - epoch, 1)
                    )
                phi = apt.phis[-1]
            else:
                phi = r
            if method == "nmfc":
                head.fit_bank(z_all, y_tr, seed=seed)
                logits_te = head.forward_bank(encoder(x_te).float())
            else:
                logits_te = head(encoder(x_te))
            top1 = topk_accuracy(logits_te, y_te, ks=(1,))["top1"]
        lr_used = optimizer.param_groups[0]["lr"]  # 이번 에폭이 실제 사용한 lr (step 전에 기록)
        if scheduler is not None and not transitioned:  # 전환 에폭엔 새 스케줄러를 step하지 않음
            scheduler.step()
        history["epoch"].append(epoch)
        history["loss"].append(float(np.mean(losses)))
        history["lr"].append(lr_used)
        history["phase"].append(phase)
        history["fisher_ratio"].append(r)
        history["phi"].append(phi)
        history["test_top1"].append(top1)

    # 최종 산출물
    encoder.eval()
    with torch.no_grad():
        z_te = encoder(x_te).float()
        sil = silhouette_of_embeddings(z_te.cpu().numpy(), y_te.cpu().numpy())
    if apt:
        history["transition_epoch"] = apt.transition_epoch
        history["transition_reason"] = apt.transition_reason

    ckpt = {"encoder": encoder.state_dict(), "spec": spec}
    if method == "softmax":
        ckpt["head"] = head.state_dict()
    torch.save(ckpt, paths["ckpt"] / "ckpt.pt")
    save_json(paths["log"] / "history.json", history)
    npz = {"z_test": z_te.cpu().numpy(), "y_test": y_te.cpu().numpy()}
    if method == "nmfc":
        npz.update(head.bank_state())
    np.savez(paths["result"] / "embeddings.npz", **npz)

    return {
        "runner": "synthetic",
        "dataset": spec["dataset"],
        "method": method,
        "test_top1": history["test_top1"][-1],
        "silhouette": sil,
        "transition_epoch": apt.transition_epoch if apt else None,
        "transition_reason": apt.transition_reason if apt else None,
        "epochs": epochs,
    }


def load_synthetic_model(name: str, device=None):
    """캐시된 합성 실험에서 (encoder, head, spec) 복원 — 결정경계/에너지 지형용."""
    device = device or get_device()
    ckpt = torch.load(exp_paths(name)["ckpt"] / "ckpt.pt", map_location="cpu", weights_only=True)
    spec = ckpt["spec"]
    encoder = EncoderMLP(2, tuple(spec["hidden"]), spec["embed_dim"])
    encoder.load_state_dict(ckpt["encoder"])
    encoder.to(device).eval()
    if spec["method"] == "softmax":
        emb = load_embeddings(name)
        n_classes = int(emb["y_test"].max()) + 1
        head = LinearHead(spec["embed_dim"], n_classes)
        head.load_state_dict(ckpt["head"])
        head.to(device).eval()
    else:
        head = NMFCHead()
        head.load_bank_state(load_embeddings(name), device=device)
    return encoder, head, spec


def synthetic_predict_fn(name: str, device=None):
    """(N,2) numpy 그리드 → 클래스 예측 numpy. viz.plot_decision_boundary용."""
    device = device or get_device()
    encoder, head, spec = load_synthetic_model(name, device)

    @torch.no_grad()
    def fn(grid: np.ndarray) -> np.ndarray:
        z = encoder(torch.from_numpy(grid).to(device)).float()
        if spec["method"] == "softmax":
            return head(z).argmax(1).cpu().numpy()
        return head.predict(z).cpu().numpy()

    return fn


def synthetic_energy_fn(name: str, device=None):
    """(N,2) numpy 그리드 → E_total. viz.plot_energy_landscape용 (NMFC 전용)."""
    device = device or get_device()
    encoder, head, spec = load_synthetic_model(name, device)
    assert spec["method"] == "nmfc"

    @torch.no_grad()
    def fn(grid: np.ndarray) -> np.ndarray:
        z = encoder(torch.from_numpy(grid).to(device)).float()
        return head.ood_score(z).cpu().numpy()

    return fn


# ---------------------------------------------------------------- 이미지 러너

def _build_image_head(method: str, spec: dict, cfg: dict, n_classes: int, device):
    m = spec["embed_dim"]
    if method in ("softmax", "deepmfa"):
        return LinearHead(m, n_classes).to(device)
    if method == "arcface":
        return ArcFaceHead(m, n_classes, s=spec.get("s", 30.0), margin=spec.get("margin", 0.2)).to(device)
    if method == "proto":
        return ProtoHead(m, n_classes).to(device)
    if method == "normsoftmax":
        return NormSoftmaxHead(m, n_classes, scale=spec.get("scale", 20.0)).to(device)
    if method == "proxyanchor":
        return ProxyAnchorHead(m, n_classes).to(device)
    if method == "nmfc":
        return make_nmfc_head(spec, cfg)
    if method in ("triplet", "ms"):  # 무헤드 손실 (임베딩만)
        return None
    raise ValueError(f"알 수 없는 method: {method}")


def _image_train_loss(method, head, z, y, spec, phase):
    if method == "softmax" or method == "proto":
        return F.cross_entropy(head(z), y)
    if method == "arcface":
        return F.cross_entropy(head(z, y), y)
    if method == "deepmfa":
        return F.cross_entropy(head(z), y) + spec.get("beta", 0.1) * mfa_regularizer(z, y)
    if method == "triplet":
        return batch_hard_triplet_loss(z, y, margin=spec.get("margin", 0.3))
    if method == "normsoftmax":
        return F.cross_entropy(head(z), y)
    if method == "ms":
        return multi_similarity_loss(z, y, alpha=spec.get("ms_alpha", 2.0),
                                     beta=spec.get("ms_beta", 50.0), base=spec.get("ms_base", 0.5),
                                     eps=spec.get("ms_eps", 0.1))
    if method == "proxyanchor":
        return proxy_anchor_loss(z, y, head.proxies, margin=spec.get("pa_margin", 0.1),
                                 alpha=spec.get("pa_alpha", 32.0))
    if method == "nmfc":
        if phase == LINEAR:
            return linear_mfa_loss(z, y)
        g, y_local = head.forward_batch(z, y)
        return nmfc_loss(g, y_local)
    raise ValueError(method)


@torch.no_grad()
def _quick_top1(method, head, encoder, eval_loader, device, bank_src=None):
    """중간 점검용 top-1. NMFC는 bank_src=(z_probe, y_probe)로 임시 뱅크를 만든다."""
    z_te, y_te = embed_loader(encoder, eval_loader, device)
    if method == "nmfc":
        if isinstance(head, VariantNMFCHead):
            head.fit_coreset_bank(bank_src[0].to(device), bank_src[1].to(device))
        else:
            head.fit_bank(bank_src[0].to(device), bank_src[1].to(device))
        logits = head.forward_bank(z_te.to(device)).cpu()
    elif method in ("triplet", "ms"):
        z_p, y_p = bank_src
        z_p, z_te = F.normalize(z_p), F.normalize(z_te)  # 학습과 같은 단위구 위에서 평가
        protos = torch.stack([z_p[y_p == c].mean(0) for c in torch.unique(y_p)])
        logits = -torch.cdist(z_te, protos).pow(2)
    else:
        logits = head(z_te.to(device)).cpu()
    return topk_accuracy(logits, y_te, ks=(1,))["top1"]


def _run_image_core(spec: dict, cfg: dict, paths: dict, seen_classes=None):
    """이미지 학습 공통 본체. (metrics, encoder, head, z_test, y_test)를 반환.

    method=="nmfc"는 variant(v3 4축)를 해석해 v1 경로(variant 없음 = 바이트 동일)와
    v3 경로(softmin/hinge/negJ/online)를 한 루프에서 처리한다. 비-nmfc 경로는 전부 불변.

    seen_classes(개방셋)가 주어지면 train·plain_train split만 해당 클래스로 서브셋한다
    (test·라벨 공간·num_classes 불변). 기본 None = 기존 경로와 바이트 동일.
    """
    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = get_device()
    root = PROJECT_ROOT / "data" / "raw"
    method = spec["method"]
    img_size = spec["img_size"]

    # ---- variant 해석·검증 (nmfc 전용; variant 없으면 V1_DEFAULTS = 바이트 동일) ----
    variant = None
    is_variant = False
    nm = {**cfg["nmfc"], **spec.get("nmfc", {})}
    if method == "nmfc":
        variant = resolve_variant(spec)
        validate_variant(variant, spec, cfg)
        is_variant = not is_v1(variant)
    online = bool(variant and variant["bank"] == "online")

    # 라벨 노이즈(plan14): 해석은 여기 한 곳 — train_ds(PK 배치)·plain_train_ds(프로브·앵커·뱅크)가
    # 같은 해석본을 받아 동일 실현 공유. label_noise 키 부재 시 None → 기존 경로와 바이트 동일.
    label_noise = resolve_label_noise(
        spec.get("label_noise"),
        {"cifar100": 100, "cifar10": 10}.get(spec["dataset"]),
        seen_classes)
    train_ds, test_ds, n_classes = get_image_datasets(
        spec["dataset"], root, img_size, seen_classes=seen_classes, label_noise=label_noise)
    plain_train_ds = get_plain_train_dataset(
        spec["dataset"], root, img_size, seen_classes=seen_classes, label_noise=label_noise)
    # 온라인 뱅크: Phase II는 표준 셔플 배치 + sample-index 태그(IndexedDataset, 이웃은 뱅크가 공급),
    # Phase I(-J/선형판별)은 이웃을 안 쓰고 산포행렬 추정에 클래스 구조가 필요하므로 PK 배치를 쓴다
    # (셔플 배치는 클래스당 ~2샘플이라 S_W가 랭크결핍 → -J 폭주로 임베딩이 붕괴함을 스모크로 확인).
    online_loader = None
    if online:
        online_loader = make_train_loader(
            IndexedDataset(train_ds), "shuffle", spec["batch_size"], spec["batch_p"],
            spec["batch_k"], spec["num_workers"], seed,
        )
        train_loader = make_train_loader(  # Phase I 전용 PK
            train_ds, "pk", spec["batch_size"], spec["batch_p"], spec["batch_k"],
            spec["num_workers"], seed,
        )
    else:
        mode = "pk" if method in PK_METHODS else "shuffle"
        train_loader = make_train_loader(
            train_ds, mode, spec["batch_size"], spec["batch_p"], spec["batch_k"],
            spec["num_workers"], seed,
        )
    test_loader = make_eval_loader(test_ds, num_workers=spec["num_workers"])

    # 고정 프로브(train, 증강 없음)와 간이 평가(test 부분집합) 로더
    rng = np.random.default_rng(seed)
    probe_size = min(cfg["apt"]["probe_size"], len(plain_train_ds))
    probe_idx = rng.choice(len(plain_train_ds), size=probe_size, replace=False)
    probe_idx_t = torch.as_tensor(probe_idx)  # 온라인 뱅크 LOO 태그(전역 인덱스)
    probe_loader = make_eval_loader(Subset(plain_train_ds, probe_idx.tolist()),
                                    num_workers=spec["num_workers"])
    sub_size = min(spec.get("eval_subset", 2000), len(test_ds))
    sub_idx = rng.choice(len(test_ds), size=sub_size, replace=False)
    evalsub_loader = make_eval_loader(Subset(test_ds, sub_idx.tolist()),
                                      num_workers=spec["num_workers"])

    encoder = ResNet50Encoder(
        spec["embed_dim"], pretrained=spec.get("pretrained", True)
    ).to(device).to(memory_format=torch.channels_last)
    if spec.get("freeze_backbone", False):  # linear-probe: 몸통 동결, 투영층(fc)+헤드만 학습
        for p in encoder.net.parameters():
            p.requires_grad = False
        for p in encoder.net.fc.parameters():
            p.requires_grad = True
    if method == "nmfc":
        head = build_nmfc_head(spec, cfg, variant)
        if isinstance(head, torch.nn.Module):
            head = head.to(device)
    else:
        head = _build_image_head(method, spec, cfg, n_classes, device)
    opt_cfg = {**cfg["optim"], **spec.get("optim", {})}  # 실험별 오버라이드 허용
    epochs = spec["epochs"]
    head_params = list(head.parameters()) if isinstance(head, torch.nn.Module) else []

    # 온라인 뱅크(전환 시 충전; Phase I은 뱅크 불필요)
    bank = None
    if online:
        assert nm["bank_q"] >= nm["k1"] + 1, f"bank_q={nm['bank_q']} < k1+1={nm['k1'] + 1}"
        bank = OnlineClassBank(n_classes, nm["bank_q"], spec["embed_dim"], device)

    if method == "nmfc":
        apt = build_warmup(spec, cfg, variant)
        # 옵티마이저는 encoder+head_params (v1은 head_params==[]라 동일; hinge_temp는 ρ 포함)
        optimizer = torch.optim.AdamW(
            list(encoder.parameters()) + head_params,
            lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"],
        )
        # NoWarmup 절제군도 동일한 cosine 예산을 받아야 lr 스케줄 오염이 없다
        if apt.phase == NONLINEAR:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        else:
            scheduler = None
    else:
        apt = None
        plr = float(spec.get("proxy_lr_mult", 1.0))  # plan11: Proxy-Anchor 공식 구현의 프록시 lr 배율
        if plr != 1.0 and head_params:
            groups = [{"params": list(encoder.parameters()), "lr": opt_cfg["lr"]},
                      {"params": head_params, "lr": opt_cfg["lr"] * plr}]
            optimizer = torch.optim.AdamW(groups, weight_decay=opt_cfg["weight_decay"])
        else:
            optimizer = torch.optim.AdamW(
                list(encoder.parameters()) + head_params,
                lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"],
            )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    apt_val = variant["apt"] if variant else None  # negJ 계열 form 분기(sphere/trace/원공간)
    kappa_ref = None  # negJ_trace: 전환 이전 J 중앙값(포화 앵커). Phase I 매 에폭 갱신(late-binding).

    def nmfc_batch_loss(z, y, idx_batch, phase, epoch):
        """nmfc 배치 손실 — Phase I(reciprocal:linear_mfa / negJ 계열:-J), Phase II(hinge/online + 램프)."""
        if apt_val == "negJ_sphere":  # v4 기본: 단위 구면 -J(척도 불변)
            phase_one = neg_j_loss(z, y, sphere=True)
        elif apt_val == "negJ_trace":  # v4 절제: 원공간 포화형 -J/(J+κ) (κ는 Phase I 러닝 중앙값)
            phase_one = neg_j_loss(z, y, kappa=kappa_ref)
        elif apt_val == "negJ":  # v3: 원공간 -J
            phase_one = neg_j_loss(z, y)
        else:  # reciprocal(v1)
            phase_one = linear_mfa_loss(z, y)
        if phase == LINEAR:
            return phase_one
        if online:
            g, y_local = head.forward_online(z, y, bank, idx_batch)
            bank.push(z.detach(), y, idx_batch)
        else:
            g, y_local = head.forward_batch(z, y)
        loss_nmfc = nmfc_loss(g, y_local)
        # 램프(전환 후 R에폭 선형 혼합)는 negJ 계열에만. 단 online은 Phase II가 셔플 배치라
        # -J(neg_j_loss)가 랭크결핍 S_W로 폭주하므로 램프를 끄고 하드 스위치한다(스모크로 확인;
        # sphere는 스케일 자유도만 닫고 랭크결핍은 못 고치므로 online 하드 스위치를 계승).
        if is_negj_family(apt_val) and not online:
            alpha = apt.alpha(epoch)
            if alpha < 1.0:
                return (1.0 - alpha) * phase_one + alpha * loss_nmfc
        return loss_nmfc

    history = _new_history()
    if is_variant:  # 변형 진단 키(v1 history는 불변 유지)
        for k in ("tau", "hinge_rate", "alpha", "j", "embed_norm"):
            history[k] = []
    eval_every = 5
    use_amp = device.type == "cuda"

    for epoch in range(1, epochs + 1):
        encoder.train()
        if isinstance(head, torch.nn.Module):
            head.train()
        phase = apt.phase if apt else "single"
        # 온라인: Phase I은 PK(train_loader), Phase II는 셔플+idx(online_loader)
        epoch_loader = online_loader if (online and phase != LINEAR) else train_loader
        online_batch = online and phase != LINEAR
        losses = []
        for batch in epoch_loader:
            if online_batch:
                x, y, idx_batch = batch
                idx_batch = idx_batch.to(device, non_blocking=True)
            else:
                x, y = batch
                idx_batch = None
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                z = encoder(x)
            z = z.float()  # 에너지/손실 계산은 fp32
            if method == "nmfc":
                loss = nmfc_batch_loss(z, y, idx_batch, phase, epoch)
            else:
                loss = _image_train_loss(method, head, z, y, spec, phase)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + head_params, opt_cfg["grad_clip"]
            )
            optimizer.step()
            losses.append(loss.item())

        # 에폭 말 처리
        encoder.eval()
        r, phi, j = None, None, None
        z_probe = y_probe = None
        transitioned = False
        if method == "nmfc":
            z_probe, y_probe = embed_loader(encoder, probe_loader, device)
            if is_negj_family(variant["apt"]):  # 모니터도 최적화 대상과 동일 기준 J
                j = fisher_j(z_probe, y_probe, sphere=(variant["apt"] == "negJ_sphere")).item()
                new_phase = apt.update(j)
                if variant["apt"] == "negJ_trace" and phase == LINEAR:
                    kappa_ref = float(np.median(apt.js))  # 전환 이전 J 러닝 중앙값(포화 앵커, 비순환)
            else:
                r = fisher_ratio(z_probe, y_probe).item()
                new_phase = apt.update(r)
            phi = apt.phis[-1]
            if new_phase == NONLINEAR and phase == LINEAR:
                transitioned = True
                # ① 온라인 뱅크 충전(전체 train 임베딩), ② 힌지 마진 앵커(프로브 E⁻_{y_i})
                if online:
                    charge_loader = make_eval_loader(plain_train_ds, num_workers=spec["num_workers"])
                    z_full, y_full = embed_loader(encoder, charge_loader, device)
                    bank.charge(z_full.to(device), y_full.to(device),
                                torch.arange(len(z_full), device=device))
                if isinstance(head, VariantNMFCHead):
                    mz, my = z_probe[:2048].to(device), y_probe[:2048].to(device)
                    midx = probe_idx_t[:2048].to(device) if online else None
                    e_neg_true = head.margin_energies(mz, my, bank=bank if online else None, idx=midx)
                    head.set_margin(e_neg_true)
                optimizer = torch.optim.AdamW(
                    list(encoder.parameters()) + head_params,
                    lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"],
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(epochs - epoch, 1)
                )
        lr_used = optimizer.param_groups[0]["lr"]  # 이번 에폭이 실제 사용한 lr (step 전에 기록)
        if scheduler is not None and not transitioned:  # 전환 에폭엔 새 스케줄러를 step하지 않음
            scheduler.step()

        top1 = None
        if epoch % eval_every == 0 or epoch == epochs:
            skip_variant = isinstance(head, VariantNMFCHead) and not bool(torch.isfinite(head.m))
            if method in ("nmfc", "triplet", "ms") and not skip_variant:
                if z_probe is None:
                    z_probe, y_probe = embed_loader(encoder, probe_loader, device)
                top1 = _quick_top1(method, head, encoder, evalsub_loader, device,
                                   bank_src=(z_probe, y_probe))
            elif method not in ("nmfc", "triplet", "ms"):
                top1 = _quick_top1(method, head, encoder, evalsub_loader, device)

        history["epoch"].append(epoch)
        history["loss"].append(float(np.mean(losses)))
        history["lr"].append(lr_used)
        history["phase"].append(phase)
        history["fisher_ratio"].append(r)
        history["phi"].append(phi)
        history["test_top1"].append(top1)
        if is_variant:  # 변형 진단 (τ 궤적·힌지 발동률·α 램프·J·임베딩 노름)
            tau_v = head.tau.item() if isinstance(head, VariantNMFCHead) else None
            embed_norm = float(z_probe.norm(dim=1).mean().item())
            hinge_rate = None
            if isinstance(head, VariantNMFCHead) and bool(torch.isfinite(head.m)):
                mz, my = z_probe[:1024].to(device), y_probe[:1024].to(device)
                midx = probe_idx_t[:1024].to(device) if online else None
                en = head.margin_energies(mz, my, bank=bank if online else None, idx=midx)
                hinge_rate = float((en >= head.m).float().mean().item())
            # 손실에 실제 적용된 램프 계수(online은 하드 스위치라 Phase II=1.0, reciprocal은 None)
            if not is_negj_family(variant["apt"]):
                eff_alpha = None
            elif phase == LINEAR:
                eff_alpha = 0.0
            elif online:
                eff_alpha = 1.0
            else:
                eff_alpha = apt.alpha(epoch)
            history["tau"].append(tau_v)
            history["hinge_rate"].append(hinge_rate)
            history["alpha"].append(eff_alpha)
            history["j"].append(j)
            history["embed_norm"].append(embed_norm)
        print(f"[{spec['name']}] epoch {epoch}/{epochs} phase={phase} "
              f"loss={np.mean(losses):.4f}" + (f" top1={top1:.4f}" if top1 else ""))

    # ---------------- 최종 평가 ----------------
    encoder.eval()
    z_test, y_test = embed_loader(encoder, test_loader, device)
    if method == "nmfc":
        full_train_loader = make_eval_loader(plain_train_ds, num_workers=spec["num_workers"])
        z_train, y_train = embed_loader(encoder, full_train_loader, device)
        if isinstance(head, VariantNMFCHead):
            head.fit_coreset_bank(z_train.to(device), y_train.to(device), seed=seed)
        else:
            head.fit_bank(z_train.to(device), y_train.to(device), seed=seed)
        logits = head.forward_bank(z_test.to(device)).cpu()
    elif method in ("triplet", "ms"):
        full_train_loader = make_eval_loader(plain_train_ds, num_workers=spec["num_workers"])
        z_train, y_train = embed_loader(encoder, full_train_loader, device)
        z_train_n, z_test_n = F.normalize(z_train), F.normalize(z_test)
        protos = torch.stack([z_train_n[y_train == c].mean(0) for c in torch.unique(y_train)])
        logits = -torch.cdist(z_test_n, protos).pow(2)
    else:
        with torch.no_grad():
            logits = torch.cat([
                head(z_test[s : s + 4096].to(device)).cpu()
                for s in range(0, len(z_test), 4096)
            ])
    accs = topk_accuracy(logits, y_test, ks=(1, 5))
    sil = silhouette_of_embeddings(z_test.numpy(), y_test.numpy())

    if apt:
        history["transition_epoch"] = apt.transition_epoch
        history["transition_reason"] = apt.transition_reason

    ckpt = {"encoder": encoder.state_dict(), "spec": spec}
    if isinstance(head, torch.nn.Module):
        ckpt["head"] = head.state_dict()
    torch.save(ckpt, paths["ckpt"] / "ckpt.pt")
    save_json(paths["log"] / "history.json", history)
    npz = {"z_test": z_test.numpy(), "y_test": y_test.numpy()}
    if method == "nmfc":
        npz.update(head.bank_state())
    np.savez(paths["result"] / "embeddings.npz", **npz)

    metrics = {
        "runner": "image",
        "dataset": spec["dataset"],
        "method": method,
        "top1": accs.get("top1"),
        "top5": accs.get("top5"),
        "silhouette": sil,
        "transition_epoch": apt.transition_epoch if apt else None,
        "epochs": epochs,
        "img_size": img_size,
    }
    if method == "nmfc":  # collect_results 프로그램적 그룹핑용 (캐시된 v1엔 없음 → 읽는 쪽이 "v1" 채움)
        metrics["variant"] = variant
        metrics["variant_code"] = variant_code(variant)
    if label_noise is not None:  # plan14 — 키 부재 시 기존 metrics 스키마 완전 불변
        metrics["label_noise"] = dict(spec["label_noise"])
        metrics["label_noise_stats"] = getattr(train_ds, "noise_stats", None)
    return metrics, encoder, head, z_test, y_test


def run_image(spec: dict, cfg: dict, paths: dict) -> dict:
    metrics, *_ = _run_image_core(spec, cfg, paths)
    return metrics


# ---------------------------------------------------------------- OOD 러너

def run_ood(spec: dict, cfg: dict, paths: dict) -> dict:
    metrics, encoder, head, z_test, y_test = _run_image_core(spec, cfg, paths)
    device = get_device()
    root = PROJECT_ROOT / "data" / "raw"
    _, ood_ds, _ = get_image_datasets(spec["ood_dataset"], root, spec["img_size"])
    ood_loader = make_eval_loader(ood_ds, num_workers=spec["num_workers"])
    z_ood, _ = embed_loader(encoder, ood_loader, device)

    method = spec["method"]
    if method == "nmfc":
        scores_in = head.ood_score(z_test.to(device)).cpu().numpy()
        scores_out = head.ood_score(z_ood.to(device)).cpu().numpy()
        score_name = "E_total"
    else:  # softmax 비교군: 1 - MSP (max softmax probability)
        with torch.no_grad():
            def msp(z):
                probs = torch.cat([
                    F.softmax(head(z[s : s + 4096].to(device)), dim=1).max(1).values.cpu()
                    for s in range(0, len(z), 4096)
                ])
                return (1.0 - probs).numpy()
            scores_in, scores_out = msp(z_test), msp(z_ood)
        score_name = "1-MSP"

    auc = ood_auc(scores_in, scores_out)
    # 기존 embeddings.npz에 OOD 점수 추가 저장
    emb_file = paths["result"] / "embeddings.npz"
    with np.load(emb_file) as z:
        npz = dict(z)
    npz.update({"ood_scores_in": scores_in, "ood_scores_out": scores_out})
    np.savez(emb_file, **npz)

    metrics.update({"runner": "ood", "ood_dataset": spec["ood_dataset"],
                    "ood_auc": auc, "ood_score": score_name})
    return metrics


# ---------------------------------------------------------------- 개방셋(open-set) 러너

def run_openset(spec: dict, cfg: dict, paths: dict) -> dict:
    """개방셋 러너 — n_seen 클래스만 학습, 전체 클래스로 평가(한 학습 → 두 평가).

    평가 A(일반화 100-way): 추론 뱅크를 전체 클래스로 fit → predict → top1(all/seen/novel).
    평가 B(near-OOD): 학습 클래스만 뱅크로 두고 ood_score=min_k E⁺ → AUC(seen vs novel test).
    softmax 대조군은 100-logit 직접(컬럼==라벨)으로 top1 + 1-MSP AUC — novel은 미학습이라 top1≈0.
    """
    os_cfg = spec["open_set"]
    total = 100 if spec["dataset"] == "cifar100" else 10
    n_seen = int(os_cfg["n_seen"])
    split_seed = int(os_cfg["split_seed"])
    assert n_seen < total, f"n_seen={n_seen} < total={total} 이어야 novel이 존재"
    assert n_seen >= spec["batch_p"], f"n_seen={n_seen} >= batch_p={spec['batch_p']} 필요(PK 샘플러)"
    seen = select_seen_classes(total, n_seen, split_seed)
    seen_list = [int(c) for c in seen]

    metrics, encoder, head, z_test, y_test = _run_image_core(
        spec, cfg, paths, seen_classes=seen)
    device = get_device()
    root = PROJECT_ROOT / "data" / "raw"
    method = spec["method"]

    y_np = y_test.numpy()
    seen_mask = np.isin(y_np, seen_list)
    novel_mask = ~seen_mask
    assert novel_mask.any(), "novel 테스트 샘플이 없음 — 분할/데이터 확인"

    if method == "nmfc":
        # 평가 B: 학습 클래스(seen)만 뱅크 — _run_image_core가 이미 seen 코어셋 뱅크로 fit해 둠
        scores = head.ood_score(z_test.to(device)).cpu().numpy()
        ood_auc_val = ood_auc(scores[seen_mask], scores[novel_mask])
        pred_seen = head.predict(
            z_test[torch.from_numpy(seen_mask)].to(device)).cpu().numpy()
        closed_seen_top1 = float((pred_seen == y_np[seen_mask]).mean())
        score_name = "E_total"
        # 평가 A: 전체 클래스(원본 id) plain-train으로 추론 뱅크 재구성 → 100-way
        # plan14: 추론 뱅크도 노이즈 라벨 세계로 구성(시스템이 가진 라벨은 noisy뿐 — 배포 현실적).
        # 같은 spec·seen이므로 학습 경로와 동일 실현(순수 함수 + 고정 시드).
        ln_full = resolve_label_noise(spec.get("label_noise"), total, seen_classes=seen)
        full_plain = get_plain_train_dataset(spec["dataset"], root, spec["img_size"],
                                             label_noise=ln_full)
        z_tr, y_tr = embed_loader(
            encoder, make_eval_loader(full_plain, num_workers=spec["num_workers"]), device)
        head.fit_coreset_bank(z_tr.to(device), y_tr.to(device), seed=cfg.get("seed", 42))
        assert len(head.bank_classes) == total, \
            f"평가 A 뱅크 클래스 {len(head.bank_classes)} != {total}"
        pred_all = head.predict(z_test.to(device)).cpu().numpy()
    else:  # softmax 대조군: 100-logit 직접
        with torch.no_grad():
            logits = torch.cat([
                head(z_test[s : s + 4096].to(device)).cpu()
                for s in range(0, len(z_test), 4096)
            ])
        pred_all = logits.argmax(1).numpy()
        scores = (1.0 - torch.softmax(logits, dim=1).max(1).values.numpy())
        ood_auc_val = ood_auc(scores[seen_mask], scores[novel_mask])
        closed_seen_top1 = float((pred_all[seen_mask] == y_np[seen_mask]).mean())
        score_name = "1-MSP"

    correct = pred_all == y_np
    metrics.update({
        "runner": "openset",
        "n_seen": n_seen,
        "split_seed": split_seed,
        "seen_classes": seen_list,
        "top1": float(correct.mean()),  # 헤드라인 = 100-way 전체 정확도(predict 기반)
        "top5": None,
        "open_top1_all": float(correct.mean()),
        "open_top1_seen": float(correct[seen_mask].mean()),
        "open_top1_novel": float(correct[novel_mask].mean()),
        "closed_seen_top1": closed_seen_top1,
        "open_ood_auc": ood_auc_val,
        "open_ood_score": score_name,
    })

    # embeddings.npz에 개방셋 산출물 추가(run_ood식 read-modify-write)
    emb_file = paths["result"] / "embeddings.npz"
    with np.load(emb_file) as z:
        npz = dict(z)
    npz.update({
        "seen_mask": seen_mask,
        "open_scores": scores,
        "open_pred": pred_all,
        "seen_classes": np.asarray(seen_list),
    })
    np.savez(emb_file, **npz)
    return metrics


# ---------------------------------------------------------------- 진입점

RUNNERS = {"synthetic": run_synthetic, "image": run_image, "ood": run_ood,
           "openset": run_openset}


def run_from_config(config_path: str | Path, only=None, force: bool = False) -> dict:
    cfg = load_config(config_path)
    runner = RUNNERS[cfg["runner"]]
    results = {}
    for exp in cfg["experiments"]:
        spec = {**cfg.get("defaults", {}), **exp}
        name = spec["name"]
        if only and name not in only:
            continue
        results[name] = run_or_load(name, lambda paths, s=spec: runner(s, cfg, paths), force=force)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    for name, m in run_from_config(args.config, only=args.only, force=args.force).items():
        print(name, {k: v for k, v in m.items() if k not in ("runner",)})


if __name__ == "__main__":
    main()
