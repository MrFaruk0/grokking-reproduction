# PROJECT_MEMORY.md
Last updated: 2026-03-18T15:26:00+03:00

## Status
Current phase: Setup
Last completed step: PROJECT_EXPLAIN.md created (all 9 sections)
Next step: src/data.py

## Environment
- GPU: [fill after Cell 2]
- PyTorch: [fill after Cell 2]
- CUDA: [fill after Cell 2]
- Python: [fill after Cell 2]

## Repository
- GitHub repo: [URL — fill in before first Colab session]
- Last commit: —

## Deviations from paper
— none yet

## Completed steps
- [2026-03-18] Scaffolded project: .gitignore, requirements.txt, PROJECT_EXPLAIN.md, PROJECT_MEMORY.md, src/data.py, src/model.py, src/train.py, src/analysis.py, src/figures.py, src/utils.py, notebooks/run_all.ipynb

## Failed attempts
— none yet

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
### Baseline
- Grokking observed: —
- Grokking epoch (test acc > 95%): —
- Final train acc: —
- Final test acc: —
- Wall time: —

### Further explorations
- Weight decay sweep: —
- p sweep: —
- Operations sweep: —
- Depth sweep: —

## Figures
- figure1_grokking_curve.png: pending
- figure2_progress_measures.png: pending
- figure3_fourier_embeddings.png: pending
- figure4_weight_norm.png: pending
- figure5_wd_sweep.png: pending
- figure6_p_sweep.png: pending
- figure7_operations.png: pending
- figure8_depth.png: pending

## Notes
— none yet
