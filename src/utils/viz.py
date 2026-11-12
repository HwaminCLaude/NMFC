"""시각화 — 전부 Figure를 반환하고, 데이터는 history/embeddings 캐시에서 온다.

그림은 학습 상태에 의존하지 않고 이 함수들만으로 그린다 (즉시 재실행 가능).
"""

import matplotlib.pyplot as plt
import numpy as np

from .misc import setup_matplotlib_korean

CLASS_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]


def _scatter_classes(ax, x, y, s=8, alpha=0.6):
    for c in np.unique(y):
        m = y == c
        ax.scatter(x[m, 0], x[m, 1], s=s, alpha=alpha, color=CLASS_COLORS[int(c) % 5],
                   label=f"클래스 {int(c)}")


def plot_dataset(x, y, title=""):
    setup_matplotlib_korean()
    fig, ax = plt.subplots(figsize=(6, 6))
    _scatter_classes(ax, x, y)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.set_aspect("equal")
    return fig


def plot_decision_boundary(predict_fn, x, y, title="", grid_res=300, pad=0.6):
    """predict_fn: (N,2) numpy → (N,) 클래스 예측. 배경을 예측 클래스로 칠한다."""
    setup_matplotlib_korean()
    x0 = np.linspace(x[:, 0].min() - pad, x[:, 0].max() + pad, grid_res)
    x1 = np.linspace(x[:, 1].min() - pad, x[:, 1].max() + pad, grid_res)
    gx, gy = np.meshgrid(x0, x1)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype("float32")
    zz = predict_fn(grid).reshape(gx.shape)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contourf(gx, gy, zz, levels=np.arange(zz.max() + 2) - 0.5,
                colors=CLASS_COLORS[: int(zz.max()) + 1], alpha=0.25)
    _scatter_classes(ax, x, y)
    ax.set_title(title)
    ax.set_aspect("equal")
    return fig


def plot_boundary_pair(pred_fn_a, pred_fn_b, x, y, title_a, title_b, grid_res=300, pad=0.6):
    """두 모델의 결정 경계를 나란히 비교 (왼쪽 vs 오른쪽)."""
    setup_matplotlib_korean()
    x0 = np.linspace(x[:, 0].min() - pad, x[:, 0].max() + pad, grid_res)
    x1 = np.linspace(x[:, 1].min() - pad, x[:, 1].max() + pad, grid_res)
    gx, gy = np.meshgrid(x0, x1)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype("float32")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, fn, title in [(axes[0], pred_fn_a, title_a), (axes[1], pred_fn_b, title_b)]:
        zz = fn(grid).reshape(gx.shape)
        ax.contourf(gx, gy, zz, levels=np.arange(zz.max() + 2) - 0.5,
                    colors=CLASS_COLORS[: int(zz.max()) + 1], alpha=0.25)
        _scatter_classes(ax, x, y)
        ax.set_title(title)
        ax.set_aspect("equal")
    return fig


def plot_energy_landscape(score_fn, x, y, title="", grid_res=200, pad=1.2):
    """score_fn: (N,2) → E_total. 낮은 곳(어두움)이 '아는 영역', 밝은 곳이 OOD."""
    setup_matplotlib_korean()
    x0 = np.linspace(x[:, 0].min() - pad, x[:, 0].max() + pad, grid_res)
    x1 = np.linspace(x[:, 1].min() - pad, x[:, 1].max() + pad, grid_res)
    gx, gy = np.meshgrid(x0, x1)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype("float32")
    e = np.log1p(score_fn(grid)).reshape(gx.shape)  # log 스케일이 지형을 잘 보여줌
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(gx, gy, e, cmap="magma", shading="auto")
    fig.colorbar(im, ax=ax, label="log(1 + E_total)")
    _scatter_classes(ax, x, y, s=6, alpha=0.8)
    ax.set_title(title)
    ax.set_aspect("equal")
    return fig


def plot_phase_history(history, title="", label=None, ax=None):
    """Fisher ratio 이동평균 Φ 곡선 + APT 전환 시점 세로선."""
    setup_matplotlib_korean()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure
    phis = history["phi"]
    ax.plot(range(1, len(phis) + 1), phis, label=label or "Φ (이동평균)")
    te = history.get("transition_epoch")
    if te:
        ax.axvline(te, color="crimson", linestyle="--", alpha=0.8)
        ax.annotate(f"위상 전환 (에폭 {te})", xy=(te, max(phis)), xytext=(te + 0.5, max(phis) * 0.9),
                    color="crimson", fontsize=9)
    ax.set_xlabel("에폭")
    ax.set_ylabel("피셔 비율 Φ")
    ax.set_title(title)
    if label:
        ax.legend()
    return fig


def plot_ablation_phis(histories: dict[str, dict], title=""):
    """APT 절제 비교: 이름 → history. Φ 곡선을 겹쳐 그린다."""
    setup_matplotlib_korean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (name, h) in enumerate(histories.items()):
        phis = h["phi"]
        color = f"C{i}"
        ax.plot(range(1, len(phis) + 1), phis, label=name, color=color)
        te = h.get("transition_epoch")
        if te:
            ax.axvline(te, color=color, linestyle=":", alpha=0.6)
    ax.set_xlabel("에폭")
    ax.set_ylabel("피셔 비율 Φ")
    ax.set_title(title)
    ax.legend(fontsize=9)
    return fig


def plot_embeddings_2d(z, y, title=""):
    setup_matplotlib_korean()
    fig, ax = plt.subplots(figsize=(6, 6))
    _scatter_classes(ax, np.asarray(z), np.asarray(y))
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    return fig


def plot_ood_hist(scores_in, scores_out, auc, title="", log_scale=True):
    setup_matplotlib_korean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    si, so = np.asarray(scores_in), np.asarray(scores_out)
    if log_scale:
        si, so = np.log1p(si), np.log1p(so)
        ax.set_xlabel("log(1 + 점수)")
    else:
        ax.set_xlabel("점수")
    bins = np.linspace(min(si.min(), so.min()), max(si.max(), so.max()), 60)
    ax.hist(si, bins=bins, alpha=0.6, label="본 적 있는 분포 (in)", color="#1f77b4")
    ax.hist(so, bins=bins, alpha=0.6, label="처음 보는 분포 (OOD)", color="#d62728")
    ax.set_ylabel("샘플 수")
    ax.set_title(f"{title} (AUC {auc:.3f})")
    ax.legend()
    return fig


def plot_variant_diagnostics(history, title=""):
    """v3 변형 진단: τ 궤적·힌지 발동률·α 램프·임베딩 노름을 2×2 서브플롯으로.

    history.json에 있는 키만 읽는다(없는 키는 안내 문구로 스킵) — reciprocal군은 α/j가 없고,
    v1은 이 함수를 호출하지 않는다. Figure 1개를 반환한다.
    """
    setup_matplotlib_korean()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    epochs = history["epoch"]
    panels = [
        ("tau", "학습 온도 τ", axes[0, 0]),
        ("hinge_rate", "힌지 발동률 (E- ≥ m 비율)", axes[0, 1]),
        ("alpha", "램프 계수 α", axes[1, 0]),
        ("embed_norm", "임베딩 노름 평균 |z|", axes[1, 1]),
    ]
    te = history.get("transition_epoch")
    for key, ylabel, ax in panels:
        vals = history.get(key)
        pairs = [(e, v) for e, v in zip(epochs, vals) if v is not None] if vals else []
        if pairs:
            ax.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", ms=3, color="#1f77b4")
        else:
            ax.text(0.5, 0.5, f"'{key}' 없음", ha="center", va="center", transform=ax.transAxes,
                    color="gray")
        if te:
            ax.axvline(te, color="crimson", linestyle="--", alpha=0.7)
        ax.set_xlabel("에폭")
        ax.set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_training_curves(histories: dict[str, dict], key="test_top1", title="", ylabel=None):
    """여러 실험의 학습 곡선 비교 (history의 epoch별 key)."""
    setup_matplotlib_korean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, h in histories.items():
        epochs = h["epoch"]
        vals = h[key]
        pairs = [(e, v) for e, v in zip(epochs, vals) if v is not None]
        if pairs:
            ax.plot([p[0] for p in pairs], [p[1] for p in pairs], label=name, marker="o", ms=3)
    ax.set_xlabel("에폭")
    ax.set_ylabel(ylabel or key)
    ax.set_title(title)
    ax.legend(fontsize=9)
    return fig
