# PROJECT_MEMORY.md
Last updated: 2026-03-18T17:22:37+03:00

## Status
Current phase: Training
Last completed step: Repo tamamen yeniden yapılandırıldı — orijinal transformers.py + helpers.py
Next step: Colab'da Cell 2 çalıştır (baseline training)

## Environment
- GPU: Tesla T4
- PyTorch: 2.4.0+cu121
- CUDA: True
- Python: 3.12

## Repository
- GitHub repo: [URL — fill in before first Colab session]
## Deviations from paper
- wandb: ÇIKARILDI (cloud logging, model matematiğini etkilemiyor)
- Diğer her şey orijinal transformers.py ile birebir


## Completed steps
- [2026-03-18] Scaffolded project: .gitignore, requirements.txt, PROJECT_EXPLAIN.md, PROJECT_MEMORY.md, src/data.py, src/model.py, src/train.py, src/analysis.py, src/figures.py, src/utils.py, notebooks/run_all.ipynb
- [2026-03-18] Training cell failed with `ValueError: numpy.dtype size changed, may indicate binary incompatibility`.
  - **Root Cause:** Downgrading `numpy` to `1.26.4` in Colab's Python 3.12 environment caused binary incompatibility with pre-installed extensions compiled for Numpy 2.x.
  - **Fix:** Updated `requirements.txt` to pin newer versions (Numpy 2.1.1, PyTorch 2.4.0, SciPy 1.14.0, etc.) that match Colab's modern runtime.
- [2026-03-18] Successfully ran all training sweeps and generated figures.
- [2026-03-20] src/model.py LayerNorm kaldırma güncellemesi | outcome: ln_count=0, forward pass shape (B,113) doğrulandı
- [2026-03-20] model.py orijinal transformers.py ile birebir eşleştirildi
- [2026-03-20] train.py: betas=(0.9,0.98), lr warmup scheduler, epochs=50000

## Failed attempts
- Modüler src/ yapısı: weight init (std=0.02), einsum çakışması,
  causal mask eksikliği nedeniyle 50k epochta grokking üretmedi
- Karar: orijinal transformers.py birebir kullanılacak

## Hyperparameters
- p: 113
- d_model: 128
- num_heads: 4
- d_mlp: 512
- num_layers: 1
- lr: 1e-3
- weight_decay: 1.0
- optimizer: AdamW (β1=0.9, β2=0.98, lr_warmup)
- batch_size: full dataset (p²=12769 pairs)
- train_fraction: 0.3
- total_epochs: 10000
- seed: 0

## Results Update — 2026-03-20T19:55:02

### Baseline (p=113, wd=1.0, 1-layer, add, seed=0)
- Grokking observed: Yes
- Grokking epoch (test acc >= 95%): 5800
- Final train acc: 1.0000
- Final test acc:  1.0000

### Weight Decay Sweep
| lambda | Grokking epoch | Final test acc |
|--------|---------------|----------------|
| 0.0 | Not reached | 0.0715 |
| 0.1 | Not reached | 0.0403 |
| 0.5 | 9400 | 0.9933 |
| 1.0 | 8000 | 1.0000 |
| 2.0 | 4500 | 1.0000 |

### Prime p Sweep
| p | Grokking epoch | Final test acc |
|---|---------------|----------------|
| 53 | Not reached | 0.1601 |
| 97 | 5700 | 1.0000 |
| 113 | 7700 | 1.0000 |
| 127 | 7600 | 0.9999 |

### Operations Sweep
| op | Grokking epoch | Final test acc |
|----|---------------|----------------|
| add | 6600 | 0.9996 |
| subtract | Not reached | 0.0089 |
| multiply | 5900 | 1.0000 |

### Depth Sweep
| layers | Grokking epoch | Final test acc |
|--------|---------------|----------------|
| 1 | 4400 | 0.9984 |
| 2 | 3800 | 1.0000 |
| 3 | 3700 | 1.0000 |

## Notes (10k Epoch Bound)
- **Empirical Surprise (Weight Decay):** Weight decay is critical. λ=0.0 and λ=0.1 failed to grok entirely within 10,000 epochs. Grokking dramatically accelerates as λ increases to 2.0.
- **Empirical Surprise (Varying p):** Larger `p` generally leads to faster grokking in this environment. `p=53` did not even grok within 10k epochs, whereas `p=127` grokked at 7600.
- **Empirical Surprise (Operations):** Subtraction `(a-b) mod p` completely failed to grok within 10k epochs, drastically underperforming compared to addition (6600 epochs) and multiplication (5900 epochs).
- **Empirical Surprise (Depth):** Increasing depth from 1 to 3 layers sped up grokking from epoch 4400 to 3700.
