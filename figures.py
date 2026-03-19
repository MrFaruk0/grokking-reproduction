"""
figures.py
Tüm figürleri matplotlib ile üretir.
Giriş: run_sweep / sweep_* fonksiyonlarının döndürdüğü history dict'leri.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch

COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
DPI = 150

def _save(fig, save_dir, stem):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f'{stem}.png', dpi=DPI, bbox_inches='tight')
    fig.savefig(save_dir / f'{stem}.pdf',          bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {save_dir}/{stem}.png/.pdf')

def _grok_line(ax, epoch):
    if epoch is not None:
        ax.axvline(epoch, color='#C44E52', linestyle='--', alpha=0.6,
                   linewidth=1.5, label=f'Grokking epoch ({epoch})')

def plot_grokking_curve(result: dict, save_dir='outputs'):
    epochs = result['epochs']
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.style.use('seaborn-v0_8-whitegrid')
    ax.plot(epochs, result['train_accs'], color=COLORS[0], lw=2, label='Train accuracy')
    ax.plot(epochs, result['test_accs'],  color=COLORS[1], lw=2, label='Test accuracy')
    _grok_line(ax, result['grokking_epoch'])
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        'Figure 1: Grokking Curve — Train vs. Test Accuracy over Training\n'
        f'Modular addition (a+b) mod {result["config"].p} · '
        f'{result["config"].num_layers}-layer Transformer · '
        f'AdamW (lr={result["config"].lr}, wd={result["config"].weight_decay})',
        fontsize=14)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, save_dir, 'figure1_grokking_curve')

def plot_weight_decay_sweep(results: dict, save_dir='outputs'):
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.style.use('seaborn-v0_8-whitegrid')
    for (wd, res), color in zip(sorted(results.items()), COLORS):
        ax.plot(res['epochs'], res['test_accs'], color=color, lw=2, label=f'λ = {wd}')
        if res['grokking_epoch']:
            ax.axvline(res['grokking_epoch'], color=color, linestyle='--', alpha=0.4, lw=1)
    ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('Test accuracy', fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Figure 5: Effect of Weight Decay on Grokking\nHigher λ → earlier grokking; λ=0 → no grokking', fontsize=14)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, save_dir, 'figure5_wd_sweep')

def plot_p_sweep(results: dict, save_dir='outputs'):
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.style.use('seaborn-v0_8-whitegrid')
    for (p, res), color in zip(sorted(results.items()), COLORS):
        ax.plot(res['epochs'], res['test_accs'], color=color, lw=2, label=f'p = {p}')
        if res['grokking_epoch']:
            ax.axvline(res['grokking_epoch'], color=color, linestyle='--', alpha=0.4, lw=1)
    ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('Test accuracy', fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Figure 6: Effect of Prime Modulus p on Grokking', fontsize=14)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, save_dir, 'figure6_p_sweep')

def plot_operations_sweep(results: dict, save_dir='outputs'):
    op_colors = {'add': COLORS[0], 'subtract': COLORS[1], 'multiply': COLORS[2]}
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.style.use('seaborn-v0_8-whitegrid')
    for op, res in results.items():
        color = op_colors.get(op, COLORS[3])
        ax.plot(res['epochs'], res['test_accs'], color=color, lw=2, label=f'op = {op}')
        if res['grokking_epoch']:
            ax.axvline(res['grokking_epoch'], color=color, linestyle='--', alpha=0.4, lw=1)
    ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('Test accuracy', fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Figure 7: Grokking Under Different Arithmetic Operations (mod 113)', fontsize=14)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, save_dir, 'figure7_operations')

def plot_depth_sweep(results: dict, save_dir='outputs'):
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.style.use('seaborn-v0_8-whitegrid')
    for (d, res), color in zip(sorted(results.items()), COLORS):
        ax.plot(res['epochs'], res['test_accs'], color=color, lw=2, label=f'{d}-layer')
        if res['grokking_epoch']:
            ax.axvline(res['grokking_epoch'], color=color, linestyle='--', alpha=0.4, lw=1)
    ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('Test accuracy', fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Figure 8: Effect of Transformer Depth on Grokking', fontsize=14)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, save_dir, 'figure8_depth')

def plot_fourier_spectrum(model, config, save_dir='outputs'):
    """W_E'nin Fourier power spectrum'unu çiz."""
    W_E = model.embed.W_E[:, :config.p].T.detach().cpu()  # (p, d_model)
    freqs = torch.fft.rfft(W_E, dim=0)
    power = (freqs.abs() ** 2).sum(dim=1).numpy()
    power_norm = power / power.max()
    freq_idx = np.arange(len(power))

    fig, ax = plt.subplots(figsize=(10, 4))
    plt.style.use('seaborn-v0_8-whitegrid')
    ax.bar(freq_idx, power_norm, color=COLORS[0], alpha=0.8, width=0.6)
    top5 = np.argsort(power)[-5:][::-1]
    for k in top5:
        ax.annotate(f'k={k}', xy=(k, power_norm[k]),
                    xytext=(k + 0.5, power_norm[k] + 0.04),
                    fontsize=9, color='#C44E52')
    ax.set_xlabel('Fourier frequency k', fontsize=12)
    ax.set_ylabel('Normalized power', fontsize=12)
    ax.set_title('Figure 3: Fourier Power Spectrum of Token Embeddings W_E\nGrokked model concentrates power in 3–5 key frequencies', fontsize=14)
    fig.tight_layout()
    _save(fig, save_dir, 'figure3_fourier_embeddings')

def plot_weight_norm(result: dict, save_dir='outputs'):
    # weight norm history result dict'te yoksa uyarı ver
    if 'weight_norms' not in result:
        print("weight_norms not in result, skipping figure 4")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    plt.style.use('seaborn-v0_8-whitegrid')
    ax.plot(result['epochs'], result['weight_norms'], color=COLORS[2], lw=2, label='Weight norm ‖θ‖₂')
    _grok_line(ax, result['grokking_epoch'])
    ax.set_xlabel('Epoch', fontsize=12); ax.set_ylabel('‖θ‖₂', fontsize=12)
    ax.set_title('Figure 4: Total Weight Norm Over Training', fontsize=14)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, save_dir, 'figure4_weight_norm')
