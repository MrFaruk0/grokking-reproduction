"""
src/figures.py
Figure generation for all 8 paper/exploration plots.

All figures follow the project figure standards:
    - figsize=(10, 4) single panel, (14, 5) multi-panel
    - dpi=150
    - Style: seaborn-v0_8-whitegrid
    - Colors: blue #4C72B0, orange #DD8452, green #55A868, red #C44E52
    - Font: title 14pt, axis labels 12pt, ticks 10pt, legend 10pt
    - Grokking epoch marked with vertical dashed line
    - Save both .png and .pdf

FILE: src/figures.py
CHANGES: initial implementation
DEPENDS ON: src/analysis.py
DEPENDED ON BY: notebooks/run_all.ipynb
"""

from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Colab/headless
import matplotlib.pyplot as plt

# ---- Style constants -------------------------------------------------------
COLORS = {
    "blue":   "#4C72B0",
    "orange": "#DD8452",
    "green":  "#55A868",
    "red":    "#C44E52",
    "purple": "#8172B3",
    "gray":   "#8C8C8C",
}
TITLE_FS  = 14
LABEL_FS  = 12
TICK_FS   = 10
LEGEND_FS = 10
DPI       = 150

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")  # older matplotlib fallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, save_dir: Path, stem: str) -> None:
    """Save figure as both .png and .pdf."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f"{stem}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(save_dir / f"{stem}.pdf",          bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_dir / stem}.png/.pdf")


def _find_grokking_epoch(
    epochs: List[int],
    test_acc: List[float],
    threshold: float = 0.95,
) -> Optional[int]:
    """Return the first epoch where test accuracy exceeds threshold, or None."""
    for ep, acc in zip(epochs, test_acc):
        if acc >= threshold:
            return ep
    return None


def _vline(ax: plt.Axes, epoch: Optional[int], label: bool = True) -> None:
    """Draw a vertical dashed line at the grokking epoch."""
    if epoch is not None:
        ax.axvline(
            epoch,
            color=COLORS["red"],
            linestyle="--",
            alpha=0.6,
            linewidth=1.5,
            label=f"Grokking epoch ({epoch})" if label else None,
        )


# ---------------------------------------------------------------------------
# Figure 1 — Grokking curve (train / test accuracy)
# ---------------------------------------------------------------------------

def plot_grokking_curve(
    history: Dict[str, List],
    save_dir: Path,
    threshold: float = 0.95,
) -> None:
    """
    Figure 1: Train and test accuracy over training epochs.

    Shows the classic grokking S-curve: train accuracy reaches 100% early,
    test accuracy lags for thousands of steps, then jumps to near-100%.

    Args:
        history:   Training history dict from train().
        save_dir:  Directory to save figures.
        threshold: Accuracy threshold for marking the grokking epoch.
    """
    epochs   = history["epoch"]
    tr_acc   = history["train_acc"]
    te_acc   = history["test_acc"]
    grok_ep  = _find_grokking_epoch(epochs, te_acc, threshold)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(epochs, tr_acc, color=COLORS["blue"],   lw=2, label="Train accuracy")
    ax.plot(epochs, te_acc, color=COLORS["orange"], lw=2, label="Test accuracy")
    _vline(ax, grok_ep)

    ax.set_xlabel("Epoch", fontsize=LABEL_FS)
    ax.set_ylabel("Accuracy", fontsize=LABEL_FS)
    ax.set_title(
        "Figure 1: Grokking Curve — Train vs. Test Accuracy over Training\n"
        "Modular addition (a+b) mod 113 · 1-layer Transformer · "
        "AdamW (lr=1e-3, wd=1.0)",
        fontsize=TITLE_FS,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.tick_params(labelsize=TICK_FS)
    ax.legend(fontsize=LEGEND_FS, loc="upper left")
    fig.tight_layout()

    _save(fig, save_dir, "figure1_grokking_curve")


# ---------------------------------------------------------------------------
# Figure 2 — Progress measures
# ---------------------------------------------------------------------------

def plot_progress_measures(
    history: Dict[str, List],
    save_dir: Path,
    threshold: float = 0.95,
) -> None:
    """
    Figure 2: The three progress measures alongside test accuracy.

    Panels: (a) Test accuracy, (b) Weight norm, (c) Fourier multiplicity.
    All share the same x-axis epoch. The grokking vertical line is drawn on
    all panels to show that the measures begin changing before the accuracy jump.

    Args:
        history:  Training history dict from train().
        save_dir: Directory to save figures.
    """
    epochs  = history["epoch"]
    te_acc  = history["test_acc"]
    w_norm  = history["weight_norm"]
    f_mult  = history["fourier_multiplicity"]
    grok_ep = _find_grokking_epoch(epochs, te_acc, threshold)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel (a) — Test accuracy
    axes[0].plot(epochs, te_acc, color=COLORS["orange"], lw=2)
    _vline(axes[0], grok_ep, label=True)
    axes[0].set_title("(a) Test Accuracy", fontsize=TITLE_FS - 1)
    axes[0].set_xlabel("Epoch", fontsize=LABEL_FS)
    axes[0].set_ylabel("Accuracy", fontsize=LABEL_FS)
    axes[0].set_ylim(-0.05, 1.05)

    # Panel (b) — Weight norm
    axes[1].plot(epochs, w_norm, color=COLORS["green"], lw=2)
    _vline(axes[1], grok_ep, label=False)
    axes[1].set_title("(b) Weight Norm ‖θ‖₂", fontsize=TITLE_FS - 1)
    axes[1].set_xlabel("Epoch", fontsize=LABEL_FS)
    axes[1].set_ylabel("‖θ‖₂", fontsize=LABEL_FS)

    # Panel (c) — Fourier multiplicity
    axes[2].plot(epochs, f_mult, color=COLORS["blue"], lw=2)
    _vline(axes[2], grok_ep, label=False)
    axes[2].set_title("(c) Fourier Multiplicity", fontsize=TITLE_FS - 1)
    axes[2].set_xlabel("Epoch", fontsize=LABEL_FS)
    axes[2].set_ylabel("# active frequencies", fontsize=LABEL_FS)

    for ax in axes:
        ax.tick_params(labelsize=TICK_FS)

    axes[0].legend(fontsize=LEGEND_FS)

    fig.suptitle(
        "Figure 2: Three Progress Measures for Grokking — All Precede the Accuracy Jump",
        fontsize=TITLE_FS, y=1.02,
    )
    fig.tight_layout()

    _save(fig, save_dir, "figure2_progress_measures")


# ---------------------------------------------------------------------------
# Figure 3 — Fourier spectrum of W_E
# ---------------------------------------------------------------------------

def plot_fourier_spectrum(
    spectrum: Dict[str, Any],
    save_dir: Path,
) -> None:
    """
    Figure 3: DFT power spectrum of the token embedding matrix W_E.

    Shows which Fourier frequencies are used by the grokked model.
    A well-grokked model concentrates power in ~3–5 key frequencies.

    Args:
        spectrum: Output of compute_fourier_spectrum() from src/analysis.py.
        save_dir: Directory to save figures.
    """
    freqs = spectrum["frequencies"]
    power = spectrum["power"]

    # Normalise for display
    power_norm = power / power.max()

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.bar(freqs, power_norm, color=COLORS["blue"], alpha=0.8, width=0.6)

    # Annotate top-5 frequencies
    top5 = np.argsort(power)[-5:][::-1]
    for k in top5:
        ax.annotate(
            f"k={freqs[k]}",
            xy=(freqs[k], power_norm[k]),
            xytext=(freqs[k] + 0.5, power_norm[k] + 0.04),
            fontsize=9,
            color=COLORS["red"],
        )

    ax.set_xlabel("Fourier frequency k", fontsize=LABEL_FS)
    ax.set_ylabel("Normalized power", fontsize=LABEL_FS)
    ax.set_title(
        "Figure 3: Fourier Power Spectrum of Token Embeddings W_E\n"
        "Grokked model concentrates power in 3–5 key frequencies",
        fontsize=TITLE_FS,
    )
    ax.tick_params(labelsize=TICK_FS)
    fig.tight_layout()

    _save(fig, save_dir, "figure3_fourier_embeddings")


# ---------------------------------------------------------------------------
# Figure 4 — Weight norm over training
# ---------------------------------------------------------------------------

def plot_weight_norm(
    history: Dict[str, List],
    save_dir: Path,
    threshold: float = 0.95,
) -> None:
    """
    Figure 4: Total weight norm ‖θ‖₂ over training.

    Tracks how weight decay compresses the model over time.
    The norm decreases as memorization (large weights) gives way to the
    compact Fourier circuit (smaller weights).

    Args:
        history:  Training history dict from train().
        save_dir: Directory to save figures.
    """
    epochs  = history["epoch"]
    w_norm  = history["weight_norm"]
    te_acc  = history["test_acc"]
    grok_ep = _find_grokking_epoch(epochs, te_acc, threshold)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(epochs, w_norm, color=COLORS["green"], lw=2, label="Weight norm ‖θ‖₂")
    _vline(ax, grok_ep)

    ax.set_xlabel("Epoch", fontsize=LABEL_FS)
    ax.set_ylabel("‖θ‖₂", fontsize=LABEL_FS)
    ax.set_title(
        "Figure 4: Total Weight Norm Over Training\n"
        "Weight decay compresses the memorization solution toward the Fourier circuit",
        fontsize=TITLE_FS,
    )
    ax.tick_params(labelsize=TICK_FS)
    ax.legend(fontsize=LEGEND_FS)
    fig.tight_layout()

    _save(fig, save_dir, "figure4_weight_norm")


# ---------------------------------------------------------------------------
# Figure 5 — Weight decay sweep
# ---------------------------------------------------------------------------

def plot_wd_sweep(
    wd_results: Dict[float, Dict],
    save_dir: Path,
    threshold: float = 0.95,
) -> None:
    """
    Figure 5: Test accuracy vs epoch for different weight decay values.

    Args:
        wd_results: Dict mapping λ values to history dicts (from sweep_weight_decay()).
        save_dir:   Directory to save figures.
    """
    color_list = [COLORS["blue"], COLORS["green"], COLORS["orange"],
                  COLORS["red"],  COLORS["purple"]]

    fig, ax = plt.subplots(figsize=(10, 4))

    for (wd, history), color in zip(sorted(wd_results.items()), color_list):
        epochs = history["epoch"]
        te_acc = history["test_acc"]
        label  = f"λ = {wd}"
        ax.plot(epochs, te_acc, color=color, lw=2, label=label)

        grok_ep = _find_grokking_epoch(epochs, te_acc, threshold)
        if grok_ep is not None:
            ax.axvline(grok_ep, color=color, linestyle="--", alpha=0.4, lw=1)

    ax.set_xlabel("Epoch", fontsize=LABEL_FS)
    ax.set_ylabel("Test accuracy", fontsize=LABEL_FS)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        "Figure 5: Effect of Weight Decay on Grokking\n"
        "Higher λ → earlier grokking; λ=0 → no grokking",
        fontsize=TITLE_FS,
    )
    ax.tick_params(labelsize=TICK_FS)
    ax.legend(fontsize=LEGEND_FS, loc="upper left")
    fig.tight_layout()

    _save(fig, save_dir, "figure5_wd_sweep")


# ---------------------------------------------------------------------------
# Figure 6 — Varying p
# ---------------------------------------------------------------------------

def plot_p_sweep(
    p_results: Dict[int, Dict],
    save_dir: Path,
    threshold: float = 0.95,
) -> None:
    """
    Figure 6: Test accuracy vs epoch for different prime moduli p.

    Args:
        p_results: Dict mapping prime p to history dicts (from sweep_prime_p()).
        save_dir:  Directory to save figures.
    """
    color_list = [COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["red"]]

    fig, ax = plt.subplots(figsize=(10, 4))

    for (p, history), color in zip(sorted(p_results.items()), color_list):
        epochs = history["epoch"]
        te_acc = history["test_acc"]
        ax.plot(epochs, te_acc, color=color, lw=2, label=f"p = {p}")

        grok_ep = _find_grokking_epoch(epochs, te_acc, threshold)
        if grok_ep is not None:
            ax.axvline(grok_ep, color=color, linestyle="--", alpha=0.4, lw=1)

    ax.set_xlabel("Epoch", fontsize=LABEL_FS)
    ax.set_ylabel("Test accuracy", fontsize=LABEL_FS)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        "Figure 6: Effect of Prime Modulus p on Grokking\n"
        "Larger p → more complex group → later grokking",
        fontsize=TITLE_FS,
    )
    ax.tick_params(labelsize=TICK_FS)
    ax.legend(fontsize=LEGEND_FS, loc="upper left")
    fig.tight_layout()

    _save(fig, save_dir, "figure6_p_sweep")


# ---------------------------------------------------------------------------
# Figure 7 — Operations comparison
# ---------------------------------------------------------------------------

def plot_operations_comparison(
    op_results: Dict[str, Dict],
    save_dir: Path,
    threshold: float = 0.95,
) -> None:
    """
    Figure 7: Test accuracy vs epoch for different arithmetic operations.

    Args:
        op_results: Dict mapping operation name to history dicts (from sweep_operations()).
        save_dir:   Directory to save figures.
    """
    op_colors = {
        "add":      COLORS["blue"],
        "subtract": COLORS["orange"],
        "multiply": COLORS["green"],
    }

    fig, ax = plt.subplots(figsize=(10, 4))

    for op, history in op_results.items():
        color  = op_colors.get(op, COLORS["gray"])
        epochs = history["epoch"]
        te_acc = history["test_acc"]
        ax.plot(epochs, te_acc, color=color, lw=2, label=f"op = {op}")

        grok_ep = _find_grokking_epoch(epochs, te_acc, threshold)
        if grok_ep is not None:
            ax.axvline(grok_ep, color=color, linestyle="--", alpha=0.4, lw=1)

    ax.set_xlabel("Epoch", fontsize=LABEL_FS)
    ax.set_ylabel("Test accuracy", fontsize=LABEL_FS)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        "Figure 7: Grokking Under Different Arithmetic Operations (mod 113)\n"
        "Add, subtract, multiply — all group operations on Z/pZ",
        fontsize=TITLE_FS,
    )
    ax.tick_params(labelsize=TICK_FS)
    ax.legend(fontsize=LEGEND_FS, loc="upper left")
    fig.tight_layout()

    _save(fig, save_dir, "figure7_operations")


# ---------------------------------------------------------------------------
# Figure 8 — Model depth comparison
# ---------------------------------------------------------------------------

def plot_depth_comparison(
    depth_results: Dict[int, Dict],
    save_dir: Path,
    threshold: float = 0.95,
) -> None:
    """
    Figure 8: Test accuracy vs epoch for different Transformer depths.

    Args:
        depth_results: Dict mapping num_layers to history dicts (from sweep_depth()).
        save_dir:      Directory to save figures.
    """
    color_list = [COLORS["blue"], COLORS["orange"], COLORS["green"]]

    fig, ax = plt.subplots(figsize=(10, 4))

    for (depth, history), color in zip(sorted(depth_results.items()), color_list):
        epochs = history["epoch"]
        te_acc = history["test_acc"]
        ax.plot(epochs, te_acc, color=color, lw=2, label=f"{depth}-layer")

        grok_ep = _find_grokking_epoch(epochs, te_acc, threshold)
        if grok_ep is not None:
            ax.axvline(grok_ep, color=color, linestyle="--", alpha=0.4, lw=1)

    ax.set_xlabel("Epoch", fontsize=LABEL_FS)
    ax.set_ylabel("Test accuracy", fontsize=LABEL_FS)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        "Figure 8: Effect of Transformer Depth on Grokking\n"
        "Does extra depth speed up or delay grokking?",
        fontsize=TITLE_FS,
    )
    ax.tick_params(labelsize=TICK_FS)
    ax.legend(fontsize=LEGEND_FS, loc="upper left")
    fig.tight_layout()

    _save(fig, save_dir, "figure8_depth")
