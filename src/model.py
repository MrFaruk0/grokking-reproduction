"""
src/model.py
Orijinal Nanda et al. (2023) transformer implementasyonu.
Kaynak: https://github.com/mechanistic-interpretability-grokking/progress-measures-paper/blob/main/transformers.py
HookPoint ve wandb bağımlılıkları çıkarıldı, geri kalan her şey birebir aynı.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    lr: float = 1e-3
    weight_decay: float = 1.0
    p: int = 113
    d_model: int = 128
    fn_name: str = 'add'
    frac_train: float = 0.3
    num_epochs: int = 50000
    seed: int = 0
    num_layers: int = 1
    batch_style: str = 'full'
    d_vocab: int = 114        # p + 1
    n_ctx: int = 3
    d_mlp: int = 512          # 4 * d_model
    num_heads: int = 4
    act_type: str = 'ReLU'
    use_ln: bool = False

    @property
    def d_head(self):
        return self.d_model // self.num_heads


class Embed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(d_model, d_vocab) / np.sqrt(d_model))

    def forward(self, x):
        return torch.einsum('dbp -> bpd', self.W_E[:, x])


class Unembed(nn.Module):
    def __init__(self, d_vocab, d_model):
        super().__init__()
        self.W_U = nn.Parameter(torch.randn(d_model, d_vocab) / np.sqrt(d_vocab))

    def forward(self, x):
        return x @ self.W_U


class PosEmbed(nn.Module):
    def __init__(self, max_ctx, d_model):
        super().__init__()
        self.W_pos = nn.Parameter(torch.randn(max_ctx, d_model) / np.sqrt(d_model))

    def forward(self, x):
        return x + self.W_pos[:x.shape[-2]]


class Attention(nn.Module):
    def __init__(self, d_model, num_heads, d_head, n_ctx):
        super().__init__()
        self.W_K = nn.Parameter(torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_Q = nn.Parameter(torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_O = nn.Parameter(torch.randn(d_model, d_head * num_heads) / np.sqrt(d_model))
        self.register_buffer('mask', torch.tril(torch.ones((n_ctx, n_ctx))))
        self.d_head = d_head

    def forward(self, x):
        k = torch.einsum('ihd,bpd->biph', self.W_K, x)
        q = torch.einsum('ihd,bpd->biph', self.W_Q, x)
        v = torch.einsum('ihd,bpd->biph', self.W_V, x)
        attn_scores_pre = torch.einsum('biph,biqp->biqp', k, q)
        attn_scores_masked = torch.tril(attn_scores_pre) - 1e10 * (1 - self.mask[:x.shape[-2], :x.shape[-2]])
        attn_matrix = F.softmax(attn_scores_masked / np.sqrt(self.d_head), dim=-1)
        z = torch.einsum('biph,biqp->biqh', v, attn_matrix)
        z_flat = einops.rearrange(z, 'b i q h -> b q (i h)')
        out = torch.einsum('df,bqf->bqd', self.W_O, z_flat)
        return out


class MLP(nn.Module):
    def __init__(self, d_model, d_mlp, act_type):
        super().__init__()
        self.W_in  = nn.Parameter(torch.randn(d_mlp, d_model) / np.sqrt(d_model))
        self.b_in  = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_model, d_mlp) / np.sqrt(d_model))
        self.b_out = nn.Parameter(torch.zeros(d_model))
        self.act_type = act_type
        assert act_type in ['ReLU', 'GeLU']

    def forward(self, x):
        x = torch.einsum('md,bpd->bpm', self.W_in, x) + self.b_in
        x = F.relu(x) if self.act_type == 'ReLU' else F.gelu(x)
        x = torch.einsum('dm,bpm->bpd', self.W_out, x) + self.b_out
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_mlp, d_head, num_heads, n_ctx, act_type):
        super().__init__()
        self.attn = Attention(d_model, num_heads, d_head, n_ctx)
        self.mlp  = MLP(d_model, d_mlp, act_type)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class Transformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.embed     = Embed(d_vocab=config.d_vocab, d_model=config.d_model)
        self.pos_embed = PosEmbed(max_ctx=config.n_ctx, d_model=config.d_model)
        self.blocks    = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                d_mlp=config.d_mlp,
                d_head=config.d_head,
                num_heads=config.num_heads,
                n_ctx=config.n_ctx,
                act_type=config.act_type,
            )
            for _ in range(config.num_layers)
        ])
        self.unembed = Unembed(d_vocab=config.d_vocab, d_model=config.d_model)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.unembed(x)
        return x

    # Aşağıdaki yardımcı metodlar analysis.py ile uyumluluk için:
    def get_embedding_matrix(self):
        """W_E'nin ilk p satırını döndür (operand tokenları)."""
        # W_E shape: (d_model, d_vocab) → transpose → (d_vocab, d_model) → ilk p satır
        return self.embed.W_E[:, :self.config.p].T.detach().cpu()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Geriye dönük uyumluluk için alias
GrokkingTransformer = Transformer
