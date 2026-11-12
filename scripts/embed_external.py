"""plan8 — 외부/오프더셸프 모델로 CIFAR-100 test 임베딩 (재학습 0).

각 모델은 자기 전처리로 임베딩 → results/ext_*/embeddings.npz (+ metrics.json) 저장.
모델별 try/except — 설치/다운로드/임포트 실패 시 스킵하고 계속(부분 성공 허용). src/ 불변.
실행: PYTHONUTF8=1 .venv\\Scripts\\python.exe scripts/embed_external.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import silhouette_of_embeddings  # noqa: E402
from src.utils.misc import get_device  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = str(ROOT / "data" / "raw")
device = get_device()

IMAGENET_TF = transforms.Compose([
    transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def cifar_test(transform):
    ds = datasets.CIFAR100(DATA, train=False, download=False, transform=transform)
    return DataLoader(ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)


@torch.no_grad()
def embed(model, loader):
    model.eval().to(device)
    zs, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            f = model(x)
        zs.append(f.float().cpu())
        ys.append(y)
    return torch.cat(zs).numpy(), torch.cat(ys).numpy()


def save(name, z, y):
    d = ROOT / "results" / name
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "embeddings.npz", z_test=z.astype(np.float32), y_test=y.astype(np.int64))
    sil = float(silhouette_of_embeddings(z, y))
    (d / "metrics.json").write_text(json.dumps(
        {"method": name.replace("ext_", ""), "source": "external",
         "silhouette": sil, "top1": None, "dim": int(z.shape[1])},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {name}: z{z.shape} sil={sil:.3f}")


def run(name, fn):
    try:
        z, y = fn()
        assert z.shape[0] == 10000, f"N={z.shape[0]} != 10000"
        assert len(np.unique(y)) == 100, f"{len(np.unique(y))} classes != 100"
        save(name, z, y)
    except Exception as e:  # noqa: BLE001 — 부분 성공 허용
        print(f"[SKIP] {name}: {type(e).__name__}: {e}")


# ---- 모델별 빌더 (각자 정규화) ----

def m_resnet50():
    from torchvision.models import ResNet50_Weights, resnet50
    w = ResNet50_Weights.IMAGENET1K_V2
    net = resnet50(weights=w)
    net.fc = nn.Identity()  # 2048-d 풀링 특징
    return embed(net, cifar_test(w.transforms()))


def m_vit_b16():
    from torchvision.models import ViT_B_16_Weights, vit_b_16
    w = ViT_B_16_Weights.IMAGENET1K_V1
    net = vit_b_16(weights=w)
    net.heads = nn.Identity()  # 768-d class token
    return embed(net, cifar_test(w.transforms()))


def m_convnext():
    from torchvision.models import ConvNeXt_Base_Weights, convnext_base
    w = ConvNeXt_Base_Weights.IMAGENET1K_V1
    net = convnext_base(weights=w)
    net.classifier[-1] = nn.Identity()  # norm+flatten 유지, 최종 Linear 제거 → 1024-d
    return embed(net, cifar_test(w.transforms()))


def m_clip():
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")

    class Wrap(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m.encode_image(x)

    return embed(Wrap(model), cifar_test(preprocess))


def m_dinov2():
    net = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")  # 384-d
    return embed(net, cifar_test(IMAGENET_TF))


if __name__ == "__main__":
    print(f"[device] {device}")
    run("ext_resnet50_frozen", m_resnet50)   # 오프라인
    run("ext_vit_b16", m_vit_b16)            # 다운로드
    run("ext_convnext_base", m_convnext)     # 다운로드
    run("ext_clip_vitb32", m_clip)           # 설치+다운로드
    run("ext_dinov2_vits14", m_dinov2)       # 인터넷
    print("done")
