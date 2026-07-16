"""Compact PNG previews for the real-cube benchmark (matplotlib, optional)."""
from __future__ import annotations

import os

import numpy as np


def _require_mpl():
    try:
        import matplotlib  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("matplotlib is required for previews; install the 'benchmark' extra") from exc
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _panel(plt, ax, arr, title, cmap="viridis", mask=None):
    a = np.array(arr, dtype=float)
    if mask is not None:
        a = np.where(mask, a, np.nan)
    im = ax.imshow(a, cmap=cmap, origin="lower")
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def render_previews(out, ct_roi, mask_roi, sel, ms_s, clean, drift, switch,
                    res_clean, res_drift, res_switch, on_wrong_switch):
    plt = _require_mpl()
    pdir = os.path.join(out, "previews")
    os.makedirs(pdir, exist_ok=True)
    L = ct_roi.shape[0]

    # 1. CT slices with source/target contours
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax, zi in zip(axes, [L // 4, L // 2, 3 * L // 4]):
        ax.imshow(ct_roi[zi], cmap="gray", origin="lower")
        ax.contour(mask_roi[zi] == sel.source_id, levels=[0.5], colors="tab:cyan", linewidths=0.7)
        ax.contour(mask_roi[zi] == sel.target_id, levels=[0.5], colors="tab:red", linewidths=0.7)
        ax.set_title(f"CT z={zi}  (S=cyan {sel.source_id}, T=red {sel.target_id})", fontsize=8)
        ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(pdir, "ct_slices.png"), dpi=110); plt.close(fig)

    # 2. reference surface: medial depth + validity + corruption regions
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    _panel(plt, axes[0], ms_s.depth, "medial depth (ROI-local)", mask=ms_s.valid)
    _panel(plt, axes[1], ms_s.valid.astype(float), "valid grid", cmap="gray")
    _panel(plt, axes[2], drift.region.astype(float), "drift patch", cmap="gray")
    _panel(plt, axes[3], switch.region.astype(float), "switch patch", cmap="gray")
    fig.tight_layout(); fig.savefig(os.path.join(pdir, "reference_surface.png"), dpi=110); plt.close(fig)

    # 3. clean diagnostics
    d = res_clean.diagnostics
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    _panel(plt, axes[0], d.confidence, "clean confidence", mask=d.valid)
    _panel(plt, axes[1], d.drift_score, "clean drift", mask=d.valid)
    _panel(plt, axes[2], d.review.astype(float), "clean review", cmap="magma", mask=d.valid)
    fig.tight_layout(); fig.savefig(os.path.join(pdir, "clean_fields.png"), dpi=110); plt.close(fig)

    # 4. drift diagnostics
    d = res_drift.diagnostics
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    _panel(plt, axes[0], d.chosen_offset, "drift chosen offset", cmap="coolwarm", mask=d.valid)
    _panel(plt, axes[1], d.drift_score, "drift score", mask=d.valid)
    _panel(plt, axes[2], drift.region.astype(float), "true drift patch", cmap="gray")
    fig.tight_layout(); fig.savefig(os.path.join(pdir, "drift_fields.png"), dpi=110); plt.close(fig)

    # 5. switch diagnostics + confusion
    d = res_switch.diagnostics
    pred = d.switch_score >= 0.5
    true = switch.region
    conf = np.zeros(true.shape, dtype=float)
    conf[pred & true] = 1      # TP
    conf[pred & ~true] = 2     # FP
    conf[~pred & true] = 3     # FN
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    _panel(plt, axes[0], d.switch_score, "switch score", mask=d.valid)
    _panel(plt, axes[1], d.confidence, "switch-surf confidence", mask=d.valid)
    _panel(plt, axes[2], d.review.astype(float), "review mask", cmap="magma", mask=d.valid)
    _panel(plt, axes[3], conf, "TP=1 FP=2 FN=3", cmap="nipy_spectral")
    fig.tight_layout(); fig.savefig(os.path.join(pdir, "switch_fields.png"), dpi=110); plt.close(fig)
    return pdir
