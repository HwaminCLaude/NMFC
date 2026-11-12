"""시드·디바이스·JSON·matplotlib 한글 폰트 공통 유틸."""

import json
import random
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed: int = 42) -> None:
    """재현성 시드 고정. cudnn.benchmark는 속도를 위해 켜 둔다 —
    실행 간 비트 단위 동일성은 보장하지 않는다 (통계적 재현성)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_matplotlib_korean() -> None:
    """한글 레이블이 네모(□)로 깨지는 것 방지."""
    import matplotlib

    # Windows(Malgun Gothic) 우선, 리눅스·Docker(NanumGothic / Noto Sans CJK KR) 폴백.
    # DejaVu Sans는 넣지 않는다 — 넣으면 Windows 출력(did_curve·gap_curve)이 바뀐다(실측).
    matplotlib.rcParams["font.family"] = ["Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def save_json(path: str | Path, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
