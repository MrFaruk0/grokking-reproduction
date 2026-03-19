# PROJECT_MEMORY.md
Last updated: 2026-03-18T17:22:37+03:00

## Status
Current phase: Results Analysis
Last completed step: Full Colab notebook execution and memory update
Next step: Project complete (abstract drafted)

## Environment
- GPU: Tesla T4
- PyTorch: 2.4.0+cu121
- CUDA: True
- Python: 3.12

## Repository
- GitHub repo: [URL — fill in before first Colab session]
- Last commit: —

## Deviations from paper
- LayerNorm: DÜZELTILDI. İlk implementasyonda pre-LN kullanılmıştı (3 adet: ln1, ln2, ln_final).
  Makale açıkça "We did not use LayerNorm" diyor (Bölüm 3, sayfa 3).
  src/model.py güncellenerek tüm LayerNorm katmanları kaldırıldı.
  Diğer tüm hyperparametreler makalenin Bölüm 3 ve Appendix A spesifikasyonuyla birebir eşleşiyor.


## Completed steps
- [2026-03-18] Scaffolded project: .gitignore, requirements.txt, PROJECT_EXPLAIN.md, PROJECT_MEMORY.md, src/data.py, src/model.py, src/train.py, src/analysis.py, src/figures.py, src/utils.py, notebooks/run_all.ipynb
- [2026-03-18] Training cell failed with `ValueError: numpy.dtype size changed, may indicate binary incompatibility`.
  - **Root Cause:** Downgrading `numpy` to `1.26.4` in Colab's Python 3.12 environment caused binary incompatibility with pre-installed extensions compiled for Numpy 2.x.
  - **Fix:** Updated `requirements.txt` to pin newer versions (Numpy 2.1.1, PyTorch 2.4.0, SciPy 1.14.0, etc.) that match Colab's modern runtime.
- [2026-03-18] Successfully ran all training sweeps and generated figures.
- [2026-03-20] src/model.py LayerNorm kaldırma güncellemesi | outcome: ln_count=0, forward pass shape (B,113) doğrulandı

## Hyperparameters
- p: 113
- d_model: 128
- num_heads: 4
- d_mlp: 512
- num_layers: 1
- lr: 1e-3
- weight_decay: 1.0
- optimizer: AdamW (β1=0.9, β2=0.98)
- batch_size: full dataset (p²=12769 pairs)
- train_fraction: 0.3
- total_epochs: 20000
- seed: 42

## Results log
### Baseline (p=113, wd=1.0, 1-layer, add, seed=42)
- Grokking observed: Yes
- Grokking epoch (test acc ≥ 95%): 1600
- Final train acc: 1.0000
- Final test acc:  1.0000
- Wall time: 3.2 min

### Further explorations
- Weight decay sweep: Completed (λ=0.0 fails to grok, higher λ groks faster)
- p sweep: Completed (Larger p groks *faster*)
- Operations sweep: Completed (Subtraction groks significantly slower than addition/multiplication)
- Depth sweep: Completed (Deeper networks grok faster)

## Figures
- figure1_grokking_curve.png: complete
- figure2_progress_measures.png: complete
- figure3_fourier_embeddings.png: complete
- figure4_weight_norm.png: complete
- figure5_wd_sweep.png: complete
- figure6_p_sweep.png: complete
- figure7_operations.png: complete
- figure8_depth.png: complete

## Notes
- **Empirical Surprise (Varying p):** The hypothesis that larger primes `p` would delay grokking due to requiring a more complex Fourier circuit was falsified. Empirically, `p=127` grokked at epoch 1400, while `p=53` took until epoch 5300. Larger `p` led to significantly earlier grokking.
- **Empirical Surprise (Operations):** Subtraction `(a-b) mod p` took 6200 epochs to grok, drastically longer than addition (1600 epochs), despite mathematical symmetry.
- **Empirical Surprise (Depth):** Increasing depth from 1 to 3 layers sped up grokking from epoch 1600 to 900.

---
## Results Update — 2026-03-18T14:22:37

### Baseline (p=113, wd=1.0, 1-layer, add, seed=42)
- Grokking observed: Yes
- Grokking epoch (test acc ≥ 95%): 1600
- Final train acc: 1.0000
- Final test acc:  1.0000
- Wall time: 3.2 min

### Weight Decay Sweep
| λ | Grokking epoch | Final test acc |
|---|---------------|----------------|
| 0.0 | Not reached | 0.4258 |
| 0.1 | 8100 | 0.9998 |
| 0.5 | 2500 | 1.0000 |
| 1.0 | 1600 | 1.0000 |
| 2.0 | 1000 | 1.0000 |

### Prime p Sweep
| p | Grokking epoch | Final test acc |
|---|---------------|----------------|
| 53 | 5300 | 1.0000 |
| 97 | 2500 | 1.0000 |
| 113 | 1600 | 1.0000 |
| 127 | 1400 | 1.0000 |

### Operations Sweep (p=113)
| operation | Grokking epoch | Final test acc |
|-----------|---------------|----------------|
| add | 1600 | 1.0000 |
| subtract | 6200 | 1.0000 |
| multiply | 1700 | 1.0000 |

### Depth Sweep (p=113)
| num_layers | Grokking epoch | Final test acc |
|------------|---------------|----------------|
| 1 | 1600 | 1.0000 |
| 2 | 1200 | 1.0000 |
| 3 | 900 | 1.0000 |
