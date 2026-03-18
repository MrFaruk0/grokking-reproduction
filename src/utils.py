"""
src/utils.py
Utility functions for project state management and reporting.

Implements:
    - update_memory(): Append results to PROJECT_MEMORY.md after all experiments
    - find_grokking_epoch(): Detect grokking epoch from a history dict
    - format_results_table(): Pretty-print a results table

FILE: src/utils.py
CHANGES: initial implementation
DEPENDS ON: —
DEPENDED ON BY: notebooks/run_all.ipynb
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


# ---------------------------------------------------------------------------
# Grokking epoch detection
# ---------------------------------------------------------------------------

def find_grokking_epoch(
    history: Dict[str, List],
    threshold: float = 0.95,
) -> Optional[int]:
    """
    Return the first epoch where test accuracy exceeds the threshold.

    Args:
        history:   Training history dict from train().
        threshold: Test accuracy threshold to define grokking (default: 0.95).

    Returns:
        The epoch number (int) of first grokking, or None if never reached.
    """
    for epoch, acc in zip(history["epoch"], history["test_acc"]):
        if acc >= threshold:
            return epoch
    return None


# ---------------------------------------------------------------------------
# PROJECT_MEMORY.md updater
# ---------------------------------------------------------------------------

def update_memory(
    history: Dict[str, List],
    wd_results: Optional[Dict[float, Dict]] = None,
    p_results: Optional[Dict[int, Dict]] = None,
    op_results: Optional[Dict[str, Dict]] = None,
    depth_results: Optional[Dict[int, Dict]] = None,
    memory_path: Path = Path("PROJECT_MEMORY.md"),
    threshold: float = 0.95,
) -> None:
    """
    Update PROJECT_MEMORY.md with completed experiment results.

    Reads the current PROJECT_MEMORY.md, updates the Results log section,
    and rewrites the file. This function is called from Cell 14 of the notebook
    after all experiments are complete.

    Args:
        history:       Baseline training history from train().
        wd_results:    Weight decay sweep results (or None if not run).
        p_results:     Prime p sweep results (or None if not run).
        op_results:    Operations sweep results (or None if not run).
        depth_results: Depth sweep results (or None if not run).
        memory_path:   Path to PROJECT_MEMORY.md.
        threshold:     Accuracy threshold for grokking epoch detection.
    """
    memory_path = Path(memory_path)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # ---- Baseline metrics ----
    grok_ep   = find_grokking_epoch(history, threshold)
    final_tr  = history["train_acc"][-1] if history["train_acc"] else None
    final_te  = history["test_acc"][-1]  if history["test_acc"]  else None
    wall_time = history.get("wall_time_seconds", None)

    lines = []
    lines.append(f"\n---\n")
    lines.append(f"## Results Update — {timestamp}\n\n")

    # Baseline
    lines.append("### Baseline (p=113, wd=1.0, 1-layer, add, seed=42)\n")
    lines.append(f"- Grokking observed: {'Yes' if grok_ep else 'No'}\n")
    lines.append(f"- Grokking epoch (test acc ≥ {threshold*100:.0f}%): "
                 f"{grok_ep if grok_ep else 'Not reached'}\n")
    if final_tr is not None:
        lines.append(f"- Final train acc: {final_tr:.4f}\n")
    if final_te is not None:
        lines.append(f"- Final test acc:  {final_te:.4f}\n")
    if wall_time is not None:
        lines.append(f"- Wall time: {wall_time / 60:.1f} min\n")

    # Weight decay sweep
    if wd_results is not None:
        lines.append("\n### Weight Decay Sweep\n")
        lines.append("| λ | Grokking epoch | Final test acc |\n")
        lines.append("|---|---------------|----------------|\n")
        for wd, h in sorted(wd_results.items()):
            ge = find_grokking_epoch(h, threshold)
            fa = h["test_acc"][-1] if h["test_acc"] else "—"
            fa_str = f"{fa:.4f}" if isinstance(fa, float) else str(fa)
            lines.append(f"| {wd} | {ge if ge else 'Not reached'} | {fa_str} |\n")

    # Prime p sweep
    if p_results is not None:
        lines.append("\n### Prime p Sweep\n")
        lines.append("| p | Grokking epoch | Final test acc |\n")
        lines.append("|---|---------------|----------------|\n")
        for p_val, h in sorted(p_results.items()):
            ge = find_grokking_epoch(h, threshold)
            fa = h["test_acc"][-1] if h["test_acc"] else "—"
            fa_str = f"{fa:.4f}" if isinstance(fa, float) else str(fa)
            lines.append(f"| {p_val} | {ge if ge else 'Not reached'} | {fa_str} |\n")

    # Operations sweep
    if op_results is not None:
        lines.append("\n### Operations Sweep (p=113)\n")
        lines.append("| operation | Grokking epoch | Final test acc |\n")
        lines.append("|-----------|---------------|----------------|\n")
        for op, h in op_results.items():
            ge = find_grokking_epoch(h, threshold)
            fa = h["test_acc"][-1] if h["test_acc"] else "—"
            fa_str = f"{fa:.4f}" if isinstance(fa, float) else str(fa)
            lines.append(f"| {op} | {ge if ge else 'Not reached'} | {fa_str} |\n")

    # Depth sweep
    if depth_results is not None:
        lines.append("\n### Depth Sweep (p=113)\n")
        lines.append("| num_layers | Grokking epoch | Final test acc |\n")
        lines.append("|------------|---------------|----------------|\n")
        for depth, h in sorted(depth_results.items()):
            ge = find_grokking_epoch(h, threshold)
            fa = h["test_acc"][-1] if h["test_acc"] else "—"
            fa_str = f"{fa:.4f}" if isinstance(fa, float) else str(fa)
            lines.append(f"| {depth} | {ge if ge else 'Not reached'} | {fa_str} |\n")

    # Append to file
    with open(memory_path, "a", encoding="utf-8") as f:
        f.writelines(lines)

    # Also update the "Last updated" timestamp at the top
    content = memory_path.read_text(encoding="utf-8")
    content = content.replace(
        content.split("\n")[1],     # old "Last updated: ..." line
        f"Last updated: {timestamp}",
    )
    memory_path.write_text(content, encoding="utf-8")

    print(f"PROJECT_MEMORY.md updated at {timestamp}")


# ---------------------------------------------------------------------------
# Pretty results table for quick console inspection
# ---------------------------------------------------------------------------

def format_results_table(
    results: Dict[Any, Dict],
    key_label: str = "Config",
    threshold: float = 0.95,
) -> str:
    """
    Format a sweep results dict as a plain-text table string.

    Args:
        results:   Dict mapping config key → history dict.
        key_label: Column header for the sweep variable.
        threshold: Grokking threshold.

    Returns:
        Formatted table as a string (print it directly).
    """
    header  = f"{'Config':>15}  {'Grokking epoch':>16}  {'Final test acc':>14}"
    divider = "-" * len(header)
    rows    = [header, divider]

    for key, h in sorted(results.items(), key=lambda x: str(x[0])):
        ge = find_grokking_epoch(h, threshold)
        fa = h["test_acc"][-1] if h["test_acc"] else float("nan")
        ge_str = str(ge) if ge is not None else "Not reached"
        rows.append(f"{str(key):>15}  {ge_str:>16}  {fa:>14.4f}")

    return "\n".join(rows)
