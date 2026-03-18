"""
src/analysis.py
Mechanistic interpretability analysis for grokking experiments.

Implements:
    - compute_fourier_spectrum(): DFT power spectrum of W_E
    - compute_fourier_multiplicity(): count of active Fourier frequencies
    - compute_restricted_loss(): loss using only top-K Fourier components
    - compute_weight_norm(): total L2 norm of model parameters

FILE: src/analysis.py
CHANGES: initial implementation
DEPENDS ON: src/model.py
DEPENDED ON BY: src/train.py, src/figures.py, notebooks/run_all.ipynb
"""

from typing import Dict, Any

import torch
import torch.nn.functional as F
import numpy as np

from src.model import GrokkingTransformer


# ---------------------------------------------------------------------------
# Fourier spectrum of the embedding matrix
# ---------------------------------------------------------------------------

def compute_fourier_spectrum(
    model: GrokkingTransformer,
    p: int,
) -> Dict[str, Any]:
    """
    Compute the Discrete Fourier Transform power spectrum of W_E.

    For each frequency k ∈ {0, 1, ..., p//2}, compute the summed power
    across all d_model embedding dimensions. This reveals which Fourier
    frequencies the model uses in its token embeddings.

    Args:
        model: Trained GrokkingTransformer.
        p:     Prime modulus (number of operand tokens).

    Returns:
        Dict with:
            "frequencies": np.ndarray of shape (p//2 + 1,) — frequency indices
            "power":       np.ndarray of shape (p//2 + 1,) — summed DFT power
            "W_E":         np.ndarray of shape (p, d_model) — raw embedding matrix
    """
    W_E = model.get_embedding_matrix()  # (p, d_model), CPU, detached

    # DFT along the token index axis (axis 0)
    # rfft returns complex array of shape (p//2 + 1, d_model)
    freqs = torch.fft.rfft(W_E, dim=0)

    # Power at each frequency summed over all embedding dimensions
    power = (freqs.abs() ** 2).sum(dim=1)  # (p//2 + 1,)

    frequencies = np.arange(power.shape[0])

    return {
        "frequencies": frequencies,
        "power":       power.numpy(),
        "W_E":         W_E.numpy(),
    }


# ---------------------------------------------------------------------------
# Fourier multiplicity (Progress Measure 2)
# ---------------------------------------------------------------------------

def compute_fourier_multiplicity(
    W_E: torch.Tensor,
    p: int,
    tau_fraction: float = 0.01,
) -> int:
    """
    Count the number of Fourier frequencies with above-threshold power.

    A fully grokked model concentrates power in ~3–5 key frequencies.
    During memorization, power is spread across many frequencies.

    Args:
        W_E:          Token embedding matrix, shape (p, d_model) or (vocab_size, d_model).
                      Only the first p rows (operand tokens) are used.
        p:            Prime modulus (number of operand tokens).
        tau_fraction: Threshold = tau_fraction × total_power. Default: 0.01 (1%).

    Returns:
        Number of frequencies (int) with power > threshold.
    """
    W_E_ops = W_E[:p].float()  # (p, d_model)

    freqs = torch.fft.rfft(W_E_ops, dim=0)          # (p//2+1, d_model)
    power = (freqs.abs() ** 2).sum(dim=1)            # (p//2+1,)

    threshold = tau_fraction * power.sum()
    return int((power > threshold).sum().item())


# ---------------------------------------------------------------------------
# Restricted loss (Progress Measure 1)
# ---------------------------------------------------------------------------

def compute_restricted_loss(
    model: GrokkingTransformer,
    data: torch.Tensor,
    p: int,
    top_k: int = 5,
    device: str = "cpu",
) -> float:
    """
    Compute cross-entropy loss using only the top-K Fourier components of W_E.

    Projects the embedding matrix onto the top-K Fourier frequencies (by total
    power), reconstructs it, temporarily swaps it into the model, evaluates
    on the given data, then restores the original embedding.

    Args:
        model:  Trained GrokkingTransformer.
        data:   Tensor of shape (N, 4) from generate_dataset(). Used as inputs.
        p:      Prime modulus.
        top_k:  Number of top frequency components to keep.
        device: Device to run computation on.

    Returns:
        Restricted cross-entropy loss (float).
    """
    model.eval()
    W_E_full = model.embedding.weight.data.clone()  # (vocab_size, d_model)
    W_E_ops  = W_E_full[:p]                         # (p, d_model)

    # Compute DFT and select top-K frequencies by power
    freqs = torch.fft.rfft(W_E_ops.float(), dim=0)              # (p//2+1, d_model)
    power = (freqs.abs() ** 2).sum(dim=1)                       # (p//2+1,)
    top_k_indices = power.topk(top_k).indices                   # (top_k,)

    # Mask: keep only top-K frequency components
    mask = torch.zeros(freqs.shape[0], device=freqs.device)
    mask[top_k_indices] = 1.0
    freqs_restricted = freqs * mask.unsqueeze(1)                # (p//2+1, d_model)

    # Reconstruct restricted embedding via inverse DFT
    W_E_restricted = torch.fft.irfft(freqs_restricted, n=p, dim=0)  # (p, d_model)

    # Reconstruct full vocab embedding (keep "=" token unchanged)
    W_E_modified = W_E_full.clone()
    W_E_modified[:p] = W_E_restricted.to(W_E_full.dtype)

    # Temporarily swap embedding weights
    model.embedding.weight.data = W_E_modified.to(device)

    try:
        inputs = data[:, :3].to(device)
        labels = data[:, 3].to(device)
        with torch.no_grad():
            logits = model(inputs)
            loss   = F.cross_entropy(logits, labels).item()
    finally:
        # Always restore original weights
        model.embedding.weight.data = W_E_full.to(device)

    return loss


# ---------------------------------------------------------------------------
# Weight norm (Progress Measure 3)
# ---------------------------------------------------------------------------

def compute_weight_norm(model: GrokkingTransformer) -> float:
    """
    Compute the total L2 norm of all model parameters.

    ‖θ‖₂ = sqrt( Σ_i ‖p_i‖_F² )

    Args:
        model: GrokkingTransformer instance.

    Returns:
        Total weight norm (float).
    """
    total_sq = sum(
        p.detach().float().norm() ** 2
        for p in model.parameters()
    )
    return float(total_sq.sqrt().item())
