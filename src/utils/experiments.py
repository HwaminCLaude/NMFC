"""실험 캐시 계약 — run_or_load.

results/<exp>/metrics.json 이 '완료 센티널'이다: 실험 함수가 모든 산출물
(checkpoints/<exp>/ckpt.pt, logs/<exp>/history.json, results/<exp>/embeddings.npz)을
쓴 뒤 마지막에 기록된다. 파일이 있으면 재학습 없이 로드하므로, 중간에 죽은
실험은 자동으로 처음부터 다시 돈다.
"""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .misc import PROJECT_ROOT, load_json, save_json


def exp_paths(name: str) -> dict[str, Path]:
    return {
        "ckpt": PROJECT_ROOT / "checkpoints" / name,
        "log": PROJECT_ROOT / "logs" / name,
        "result": PROJECT_ROOT / "results" / name,
    }


def run_or_load(name: str, fn, force: bool = False) -> dict:
    """fn(paths)가 산출물을 쓰고 metrics dict를 돌려준다. 캐시가 있으면 로드."""
    paths = exp_paths(name)
    metrics_file = paths["result"] / "metrics.json"
    if metrics_file.exists():
        if not force:
            return load_json(metrics_file)
        metrics_file.unlink()  # 재실행이 중간에 죽어도 옛 센티널이 새 산출물과 섞이지 않게
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    metrics = fn(paths)
    metrics["name"] = name
    save_json(metrics_file, metrics)  # 완료 센티널 — 반드시 마지막
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def collect_results(names: list[str]) -> pd.DataFrame:
    """실험 이름 목록 → 지표 비교 DataFrame (없는 실험은 건너뜀)."""
    rows = []
    for n in names:
        f = exp_paths(n)["result"] / "metrics.json"
        if f.exists():
            rows.append(load_json(f))
    return pd.DataFrame(rows).set_index("name") if rows else pd.DataFrame()


def load_history(name: str) -> dict:
    return load_json(exp_paths(name)["log"] / "history.json")


def load_embeddings(name: str) -> dict:
    with np.load(exp_paths(name)["result"] / "embeddings.npz", allow_pickle=False) as z:
        return dict(z)
