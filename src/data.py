"""
src/data.py
Dataset generation for modular arithmetic grokking experiments.

Implements the exact dataset from Nanda et al. (2023):
    Task: predict (a + b) mod p
    Input: token sequence [a, b, =]
    Target: result token (a + b) mod p

FILE: src/data.py
CHANGES: initial implementation
DEPENDS ON: —
DEPENDED ON BY: src/train.py, notebooks/run_all.ipynb
"""

import torch
import random
from pathlib import Path
from typing import Tuple, List, Optional


# ---------------------------------------------------------------------------
# Core dataset functions
# ---------------------------------------------------------------------------

def generate_all_pairs(p: int, operation: str = "add") -> List[Tuple[int, int, int]]:
    """
    Generate all p² ordered pairs with their results.

    Args:
        p: Prime modulus. Operands are drawn from {0, ..., p-1}.
        operation: One of "add", "subtract", "multiply".

    Returns:
        List of (a, b, result) tuples where result = op(a, b) mod p.
    """
    pairs = []
    for a in range(p):
        for b in range(p):
            if operation == "add":
                result = (a + b) % p
            elif operation == "subtract":
                result = (a - b) % p
            elif operation == "multiply":
                result = (a * b) % p
            else:
                raise ValueError(f"Unknown operation: {operation!r}. "
                                 f"Choose from 'add', 'subtract', 'multiply'.")
            pairs.append((a, b, result))
    return pairs


def generate_dataset(
    p: int = 113,
    train_fraction: float = 0.3,
    seed: int = 42,
    operation: str = "add",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate train and test datasets for modular arithmetic.

    The dataset consists of all p² ordered pairs (a, b) with their results.
    train_fraction of pairs are used for training; the rest are held out as test.

    Token encoding:
        - Operands a, b ∈ {0, ..., p-1} → tokens 0..p-1
        - The "=" sign → token p  (vocab_size = p + 1)

    Input shape:  (N, 3)  — columns: [a, b, =]
    Target shape: (N,)    — integer in {0, ..., p-1}

    Args:
        p:              Prime modulus (paper default: 113).
        train_fraction: Fraction of all pairs used for training (paper: 0.3).
        seed:           Random seed for reproducibility (paper: 42).
        operation:      Arithmetic operation ("add", "subtract", "multiply").

    Returns:
        train_data: Tensor of shape (N_train, 4) — columns [a, b, eq_token, label]
        test_data:  Tensor of shape (N_test,  4) — columns [a, b, eq_token, label]

    Note: The 4th column (index 3) is the label. The first 3 columns are model inputs.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    all_pairs = generate_all_pairs(p, operation)
    eq_token = p  # the "=" sign is token p

    # Build input sequences and labels
    data = []
    for a, b, result in all_pairs:
        # Input: [a, b, =] as token indices; label: result
        data.append([a, b, eq_token, result])

    # Shuffle and split
    random.shuffle(data)
    n_train = int(len(data) * train_fraction)

    train_data = torch.tensor(data[:n_train], dtype=torch.long)
    test_data  = torch.tensor(data[n_train:], dtype=torch.long)

    return train_data, test_data


def get_inputs_and_labels(
    data: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split a data tensor into model inputs and labels.

    Args:
        data: Tensor of shape (N, 4) — [a, b, eq_token, label].

    Returns:
        inputs: (N, 3) — the first 3 columns [a, b, eq_token]
        labels: (N,)  — the last column (result token)
    """
    return data[:, :3], data[:, 3]


def describe_dataset(train_data: torch.Tensor, test_data: torch.Tensor, p: int) -> None:
    """
    Print a summary of the generated dataset.

    Args:
        train_data: Training data tensor of shape (N_train, 4).
        test_data:  Test data tensor of shape (N_test, 4).
        p:          Prime modulus used.
    """
    n_total = len(train_data) + len(test_data)
    print(f"Dataset Summary")
    print(f"  Modulus p       : {p}")
    print(f"  Total pairs     : {n_total}  (= p² = {p}² = {p*p})")
    print(f"  Train pairs     : {len(train_data)}  "
          f"({100 * len(train_data) / n_total:.1f}%)")
    print(f"  Test pairs      : {len(test_data)}  "
          f"({100 * len(test_data) / n_total:.1f}%)")
    print(f"  Vocab size      : {p + 1}  (tokens 0..{p-1} + '=' token {p})")
    print(f"  Input shape     : {list(train_data[:, :3].shape)}")
    print(f"  Label range     : [0, {p - 1}]")
