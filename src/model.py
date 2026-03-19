import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with pre-LayerNorm applied outside this module.

    Args:
        d_model:    Model embedding dimension.
        num_heads:  Number of attention heads. d_model must be divisible by num_heads.
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Projection matrices (no bias, following Nanda et al.)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        B, T, _ = x.shape
        H, D = self.num_heads, self.head_dim

        Q = self.W_Q(x).view(B, T, H, D).transpose(1, 2)  # (B, H, T, D)
        K = self.W_K(x).view(B, T, H, D).transpose(1, 2)
        V = self.W_V(x).view(B, T, H, D).transpose(1, 2)

        scale = math.sqrt(D)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, H, T, T)
        attn_weights = F.softmax(attn_scores, dim=-1)

        out = torch.matmul(attn_weights, V)             # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)  # (B, T, d_model)
        return self.W_O(out)


# ---------------------------------------------------------------------------
# MLP sublayer
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """
    Two-layer MLP with ReLU activation (exact paper spec).

    Pre-LayerNorm is applied outside this module.

    Args:
        d_model: Input/output dimension.
        d_mlp:   Hidden dimension (paper: 512 = 4 × 128).
    """

    def __init__(self, d_model: int, d_mlp: int) -> None:
        super().__init__()
        self.W_in  = nn.Linear(d_model, d_mlp, bias=True)
        self.W_out = nn.Linear(d_mlp, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        return self.W_out(F.relu(self.W_in(x)))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Single transformer block with pre-LayerNorm.

    Computation (following Nanda et al. §A.1):
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

    Args:
        d_model:   Embedding dimension.
        num_heads: Number of attention heads.
        d_mlp:     MLP hidden dimension.
    """

    def __init__(self, d_model: int, num_heads: int, d_mlp: int) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.mlp  = MLP(d_model, d_mlp)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class GrokkingTransformer(nn.Module):
    """
    1-layer transformer for modular arithmetic (Nanda et al. 2023).

    Architecture:
        - Token embedding W_E: (vocab_size, d_model)
        - Positional embedding W_pos: (seq_len=3, d_model)
        - TransformerBlock × num_layers
        - No LayerNorm
        - Unembedding W_U: (d_model, p) — only p output logits (not vocab_size)

    Prediction:
        Only the hidden state at position 2 (the "=" token) is unembedded.
        This follows the paper exactly: the model maps [a, b, =] → logit for (a+b mod p).

    Args:
        p:          Prime modulus. Operands ∈ {0..p-1}, vocab_size = p+1.
        d_model:    Embedding dimension (paper: 128).
        num_heads:  Attention heads (paper: 4).
        d_mlp:      MLP hidden dim (paper: 512).
        num_layers: Transformer depth (paper baseline: 1).
    """

    def __init__(
        self,
        p: int = 113,
        d_model: int = 128,
        num_heads: int = 4,
        d_mlp: int = 512,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.p = p
        self.d_model = d_model
        self.vocab_size = p + 1  # tokens 0..p-1 plus the "=" token at index p
        self.seq_len = 3         # [a, b, =]

        # Embeddings
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.pos_embedding = nn.Embedding(self.seq_len, d_model)

        # Transformer layers
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_mlp)
            for _ in range(num_layers)
        ])


        # Unembedding — output logits over p result tokens only (not the "=" token)
        self.unembed = nn.Linear(d_model, p, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with small normal noise (following Nanda et al.)."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            tokens: (batch, 3) — integer token indices [a, b, eq_token]

        Returns:
            logits: (batch, p) — un-normalized scores for each result token 0..p-1
        """
        B, T = tokens.shape
        positions = torch.arange(T, device=tokens.device).unsqueeze(0)  # (1, T)

        x = self.embedding(tokens) + self.pos_embedding(positions)  # (B, T, d_model)

        for block in self.blocks:
            x = block(x)         # (B, T, d_model)
        x = x[:, -1, :]               # (B, d_model) — last position ("=" token)
        logits = self.unembed(x)      # (B, p)
        return logits

    def get_embedding_matrix(self) -> torch.Tensor:
        """
        Return the token embedding matrix for the p operand tokens (not the "=" token).

        Returns:
            W_E: (p, d_model) on CPU as a detached tensor.
        """
        return self.embedding.weight[:self.p].detach().cpu()

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(param.numel() for param in self.parameters() if param.requires_grad)
