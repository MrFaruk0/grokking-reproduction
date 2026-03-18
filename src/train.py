"""
src/train.py
Training loop and sweep utilities for grokking experiments.

Implements:
    - train(): main training loop with full-batch AdamW (exact paper hyperparams)
    - sweep_weight_decay(): vary λ ∈ {0.0, 0.1, 0.5, 1.0, 2.0}
    - sweep_prime_p(): vary p ∈ {53, 97, 113, 127}
    - sweep_operations(): vary operation ∈ {"add", "subtract", "multiply"}
    - sweep_depth(): vary num_layers ∈ {1, 2, 3}

FILE: src/train.py
CHANGES: initial implementation
DEPENDS ON: src/data.py, src/model.py, src/analysis.py
DEPENDED ON BY: notebooks/run_all.ipynb
"""

import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.data import generate_dataset, get_inputs_and_labels
from src.model import GrokkingTransformer
from src.analysis import compute_fourier_multiplicity, compute_weight_norm


# ---------------------------------------------------------------------------
# Accuracy helper
# ---------------------------------------------------------------------------

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute top-1 accuracy.

    Args:
        logits: (N, p) un-normalized scores.
        labels: (N,) ground-truth class indices.

    Returns:
        Fraction of correct predictions ∈ [0, 1].
    """
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    model: GrokkingTransformer,
    train_data: torch.Tensor,
    test_data: torch.Tensor,
    epochs: int = 20_000,
    lr: float = 1e-3,
    weight_decay: float = 1.0,
    beta1: float = 0.9,
    beta2: float = 0.98,
    log_every: int = 100,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_every: int = 1000,
    log_path: Optional[Path] = None,
    device: Optional[str] = None,
) -> Dict[str, List]:
    """
    Train the GrokkingTransformer on modular arithmetic.

    Full-batch training: all training pairs are used in every step (no mini-batches).
    This matches the paper exactly (batch_size = full dataset).

    Args:
        model:           GrokkingTransformer instance.
        train_data:      (N_train, 4) tensor from generate_dataset().
        test_data:       (N_test, 4) tensor from generate_dataset().
        epochs:          Total training steps (paper: 20000).
        lr:              Learning rate (paper: 1e-3).
        weight_decay:    L2 regularization coefficient (paper: 1.0).
        beta1:           AdamW beta1 (paper: 0.9).
        beta2:           AdamW beta2 (paper: 0.98).
        log_every:       Log metrics every this many epochs.
        checkpoint_dir:  If provided, save model checkpoints here.
        checkpoint_every: Save checkpoint every this many epochs.
        log_path:        If provided, write metrics to this CSV file.
        device:          "cuda", "cpu", or None (auto-detect).

    Returns:
        history: Dict with keys:
            "epoch", "train_loss", "test_loss",
            "train_acc", "test_acc",
            "weight_norm", "fourier_multiplicity"
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
    )

    # Move data to device
    train_inputs, train_labels = get_inputs_and_labels(train_data)
    test_inputs,  test_labels  = get_inputs_and_labels(test_data)
    train_inputs = train_inputs.to(device)
    train_labels = train_labels.to(device)
    test_inputs  = test_inputs.to(device)
    test_labels  = test_labels.to(device)

    # Output setup
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

    history: Dict[str, List] = {
        "epoch": [],
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
        "weight_norm": [],
        "fourier_multiplicity": [],
    }

    csv_file = None
    csv_writer = None
    if log_path is not None:
        csv_file = open(log_path, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=list(history.keys()))
        csv_writer.writeheader()

    start_time = time.time()

    try:
        pbar = tqdm(range(1, epochs + 1), desc="Training", unit="epoch")
        for epoch in pbar:
            # ---- Train step ----
            model.train()
            optimizer.zero_grad()
            train_logits = model(train_inputs)               # (N_train, p)
            train_loss   = F.cross_entropy(train_logits, train_labels)
            train_loss.backward()
            optimizer.step()

            # ---- Logging ----
            if epoch % log_every == 0 or epoch == 1:
                model.eval()
                with torch.no_grad():
                    test_logits = model(test_inputs)
                    test_loss   = F.cross_entropy(test_logits, test_labels)
                    tr_acc  = accuracy(train_logits, train_labels)
                    te_acc  = accuracy(test_logits,  test_labels)
                    w_norm  = compute_weight_norm(model)
                    W_E     = model.get_embedding_matrix()
                    f_mult  = compute_fourier_multiplicity(W_E, model.p)

                row = {
                    "epoch":               epoch,
                    "train_loss":          train_loss.item(),
                    "test_loss":           test_loss.item(),
                    "train_acc":           tr_acc,
                    "test_acc":            te_acc,
                    "weight_norm":         w_norm,
                    "fourier_multiplicity": f_mult,
                }

                for key, val in row.items():
                    history[key].append(val)

                if csv_writer is not None:
                    csv_writer.writerow(row)
                    csv_file.flush()

                pbar.set_postfix({
                    "tr_acc": f"{tr_acc:.3f}",
                    "te_acc": f"{te_acc:.3f}",
                    "loss":   f"{train_loss.item():.4f}",
                })

            # ---- Checkpoint ----
            if checkpoint_dir is not None and epoch % checkpoint_every == 0:
                ckpt_path = checkpoint_dir / f"checkpoint_epoch{epoch:06d}.pt"
                torch.save({
                    "epoch":        epoch,
                    "model_state":  model.state_dict(),
                    "optim_state":  optimizer.state_dict(),
                    "history":      history,
                }, ckpt_path)

    finally:
        if csv_file is not None:
            csv_file.close()

    wall_time = time.time() - start_time
    history["wall_time_seconds"] = wall_time
    print(f"\nTraining complete. Wall time: {wall_time / 60:.1f} min")
    print(f"Final train acc: {history['train_acc'][-1]:.4f}  |  "
          f"test acc: {history['test_acc'][-1]:.4f}")

    return history


# ---------------------------------------------------------------------------
# Sweep: weight decay
# ---------------------------------------------------------------------------

def sweep_weight_decay(
    weight_decays: List[float],
    p: int = 113,
    epochs: int = 20_000,
    seed: int = 42,
    log_dir: Optional[Path] = None,
    d_model: int = 128,
    num_heads: int = 4,
    d_mlp: int = 512,
) -> Dict[float, Dict]:
    """
    Train one model per weight_decay value.

    Args:
        weight_decays: List of λ values to sweep (e.g. [0.0, 0.1, 0.5, 1.0, 2.0]).
        p:             Prime modulus.
        epochs:        Training steps per model.
        seed:          Random seed.
        log_dir:       Directory to save per-run CSVs.
        d_model, num_heads, d_mlp: Model hyperparameters (kept fixed).

    Returns:
        Dict mapping each λ value to its training history dict.
    """
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    train_data, test_data = generate_dataset(p=p, seed=seed)

    for wd in weight_decays:
        print(f"\n{'='*60}")
        print(f"Weight decay sweep: λ = {wd}")
        print(f"{'='*60}")
        torch.manual_seed(seed)
        model = GrokkingTransformer(p=p, d_model=d_model,
                                    num_heads=num_heads, d_mlp=d_mlp)
        log_path = log_dir / f"wd_{wd:.2f}.csv" if log_dir else None
        history = train(
            model, train_data, test_data,
            epochs=epochs, weight_decay=wd,
            log_path=log_path,
        )
        results[wd] = history

    return results


# ---------------------------------------------------------------------------
# Sweep: prime p
# ---------------------------------------------------------------------------

def sweep_prime_p(
    primes: List[int],
    epochs: int = 20_000,
    seed: int = 42,
    log_dir: Optional[Path] = None,
    d_model: int = 128,
    num_heads: int = 4,
    d_mlp: int = 512,
    weight_decay: float = 1.0,
) -> Dict[int, Dict]:
    """
    Train one model per prime modulus p.

    Args:
        primes:      List of prime values (e.g. [53, 97, 113, 127]).
        epochs:      Training steps per model.
        seed:        Random seed.
        log_dir:     Directory to save per-run CSVs.
        d_model, num_heads, d_mlp, weight_decay: Fixed hyperparameters.

    Returns:
        Dict mapping each p to its training history dict.
    """
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for prime in primes:
        print(f"\n{'='*60}")
        print(f"Prime sweep: p = {prime}")
        print(f"{'='*60}")
        torch.manual_seed(seed)
        train_data, test_data = generate_dataset(p=prime, seed=seed)
        model = GrokkingTransformer(p=prime, d_model=d_model,
                                    num_heads=num_heads, d_mlp=d_mlp)
        log_path = log_dir / f"p_{prime}.csv" if log_dir else None
        history = train(
            model, train_data, test_data,
            epochs=epochs, weight_decay=weight_decay,
            log_path=log_path,
        )
        results[prime] = history

    return results


# ---------------------------------------------------------------------------
# Sweep: arithmetic operations
# ---------------------------------------------------------------------------

def sweep_operations(
    operations: List[str],
    p: int = 113,
    epochs: int = 20_000,
    seed: int = 42,
    log_dir: Optional[Path] = None,
    d_model: int = 128,
    num_heads: int = 4,
    d_mlp: int = 512,
    weight_decay: float = 1.0,
) -> Dict[str, Dict]:
    """
    Train one model per arithmetic operation.

    Args:
        operations:  List of operations, e.g. ["add", "subtract", "multiply"].
        p:           Prime modulus.
        epochs:      Training steps per model.
        seed:        Random seed.
        log_dir:     Directory to save per-run CSVs.
        Others:      Fixed hyperparameters.

    Returns:
        Dict mapping each operation name to its training history dict.
    """
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for op in operations:
        print(f"\n{'='*60}")
        print(f"Operation sweep: {op} mod {p}")
        print(f"{'='*60}")
        torch.manual_seed(seed)
        train_data, test_data = generate_dataset(p=p, seed=seed, operation=op)
        model = GrokkingTransformer(p=p, d_model=d_model,
                                    num_heads=num_heads, d_mlp=d_mlp)
        log_path = log_dir / f"op_{op}.csv" if log_dir else None
        history = train(
            model, train_data, test_data,
            epochs=epochs, weight_decay=weight_decay,
            log_path=log_path,
        )
        results[op] = history

    return results


# ---------------------------------------------------------------------------
# Sweep: model depth
# ---------------------------------------------------------------------------

def sweep_depth(
    depths: List[int],
    p: int = 113,
    epochs: int = 20_000,
    seed: int = 42,
    log_dir: Optional[Path] = None,
    d_model: int = 128,
    num_heads: int = 4,
    d_mlp: int = 512,
    weight_decay: float = 1.0,
) -> Dict[int, Dict]:
    """
    Train one model per transformer depth (number of layers).

    Args:
        depths:      List of layer counts (e.g. [1, 2, 3]).
        p:           Prime modulus.
        epochs:      Training steps per model.
        seed:        Random seed.
        log_dir:     Directory to save per-run CSVs.
        Others:      Fixed hyperparameters.

    Returns:
        Dict mapping each depth to its training history dict.
    """
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    train_data, test_data = generate_dataset(p=p, seed=seed)

    for depth in depths:
        print(f"\n{'='*60}")
        print(f"Depth sweep: num_layers = {depth}")
        print(f"{'='*60}")
        torch.manual_seed(seed)
        model = GrokkingTransformer(p=p, d_model=d_model,
                                    num_heads=num_heads, d_mlp=d_mlp,
                                    num_layers=depth)
        log_path = log_dir / f"depth_{depth}.csv" if log_dir else None
        history = train(
            model, train_data, test_data,
            epochs=epochs, weight_decay=weight_decay,
            log_path=log_path,
        )
        results[depth] = history

    return results
