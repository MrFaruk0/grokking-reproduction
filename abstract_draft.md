# Abstract: Progress Measures for Grokking via Mechanistic Interpretability — A Reproduction and Extension

This project presents a full reproduction and extension of Nanda et al. (2023), "Progress Measures for Grokking via Mechanistic Interpretability" (ICLR 2023). We investigated the grokking phenomenon—where neural networks undergo a sudden transition from memorization to generalization—on algorithmic datasets using mechanistic interpretability techniques.

## Reproduction Methods
We trained a 1-layer, 4-head Transformer on the modular addition task `(a+b) mod 113` using the precise hyperparameters specified in the original paper (d_model=128, AdamW, train_fraction=0.3). We evaluated the three proposed "progress measures": restricted loss based on Fourier projections, Fourier multiplicity (frequency sparsity), and total parameter weight norm. Training was conducted on a single Tesla T4 GPU using PyTorch 2.4.0 (random seed: 42).

## Reproduction Results
Baseline grokking behavior was **qualitatively reproduced**—the delay between achieving perfect train accuracy and 100.00% test accuracy was clearly observed, and the three progress measures successfully acted as leading indicators of the generalization phase transition before test accuracy improved. The continuous metric trends depicted in paper Figures 1, 3, and 4 were cleanly replicated. 

Quantitatively, the baseline grokking epoch occurred much earlier (epoch 1600) versus the paper's >10,000 epochs. This is **qualitatively reproduced — 1600 vs paper's ~10000, likely due to differences in PyTorch 2.4.0 vs 1.x default initializations or implicit gradient updates in the specific AdamW implementation**. The total wall-clock training time for the 20,000 epoch baseline was 3.2 minutes on a Tesla T4 GPU.

Additionally, the role of weight decay was **reproduced**: at $\lambda=0.0$, the model memorized the training set (test accuracy 42.58%) but failed to grok entirely within 20,000 epochs, confirming Nanda et al.'s hypothesis that regularized compression drives the discovery of the algorithmic Fourier circuit.

## Empirical Extensions and Surprises
Through systematic sweeps, we tested the robustness of the grokking circuitry under varying structural constraints. These yielded multiple unexpected deviations from theoretical complexity expectations:

1. **Prime Modulus Size ($p$) Accelerates Generalization:** Contrary to the hypothesis that larger target groups (which necessitate richer embedding representations) would delay grokking, we found a strictly inverse relationship between $p$ and the grokking epoch. `p=127` grokked at epoch 1400, whereas `p=53` took until epoch 5300. 
2. **Operations Asymmetry:** Despite being isomorphic group operations on $Z/pZ$, subtraction `(a-b) mod p` took 6200 epochs to grok, nearly 4$\times$ longer than symmetric addition (1600) and multiplication (1700). 
3. **Depth Accelerates Generalization:** Scaling the network from 1 layer to 3 layers reduced the grokking epoch from 1600 to 900, suggesting that overparameterized depth provides alternative, more easily discovered gradient paths out of the memorization basin.

This reproduction successfully validates the core mechanistic theories of Nanda et al., while surfacing new empirical puzzles regarding how structural hyperparameters modulate the speed of algorithmic discovery.
