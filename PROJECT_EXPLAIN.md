# PROJECT_EXPLAIN.md
Complete conceptual reference for this project.

---

## 1. What is Grokking?

**Grokking** is the phenomenon where a neural network, after a long period of apparent
overfitting (training accuracy ≈ 100%, test accuracy ≈ random chance), suddenly transitions
to nearly perfect generalization — often thousands of gradient steps after first achieving
perfect training accuracy.

The term was coined by Power et al. (2022), who observed it on small algorithmic datasets
such as modular arithmetic, permutation groups, and other structured tasks. The core
observation is arresting because it contradicts the conventional view that overfitting is a
terminal state: given sufficient training time and regularization, the model can escape it.

**Why does the model memorize first?**
At initialization, the loss landscape strongly favors memorization. The memorizing solution
— essentially a giant lookup table implemented via large, sparse weight matrices — has low
curvature around itself and is found rapidly by gradient descent. The generalizing solution
requires learning a compact algorithm (see §6). This solution exists in the weight space but
is farther from initialization, and its basin of attraction is not reached early in training.
The model "finds what works first," which is memorization.

**What physically changes at the grokking point?**
The regularization pressure from weight decay (§7) continuously penalizes the large weights
required by the memorization solution. Eventually, the memorization solution's cost (in terms
of weight magnitude) exceeds that of the generalizing solution. At this tipping point, gradient
descent rapidly reorganizes the weights from the memorization regime into the generalizing
regime. This causes the abrupt jump in test accuracy.

**Connection to classical generalization theory:**
Standard generalization bounds (VC dimension, Rademacher complexity) predict generalization
from the training loss and model capacity alone — they do not directly explain the delayed
generalization seen in grokking. Grokking is a failure mode of these bounds: the model is
correctly generalized (low true risk) but requires far more training steps than theory suggests.
Nanda et al. (2023) argue that *algorithmic efficiency* — the compactness of the learned
algorithm — is what drives generalization, not sample complexity in the traditional sense.

---

## 2. What is Mechanistic Interpretability?

**Mechanistic interpretability** is the subfield of AI safety and ML interpretability that
aims to reverse-engineer a neural network into human-readable, algorithmic descriptions of what
the network computes. Rather than explaining "which input features matter" (saliency maps,
LIME, SHAP), mechanistic interpretability asks: *what algorithm is implemented in these weights?*

A key concept is the **circuit**: a subgraph of the network's weight matrices that collectively
implements a specific sub-computation. Circuits can be described mathematically. For example,
"the induction heads circuit in GPT-2 implements prefix matching" is a mechanistic claim.

**How Nanda et al. apply it to grokking:**
Rather than treating the grokking transition as a black-box phenomenon, Nanda et al. reverse-
engineer the trained model and show that:
1. The model learns to embed tokens in Fourier space (§5).
2. The attention layer computes products of Fourier features.
3. The MLP combines these products to produce `cos(ω(a+b))` directions.
4. The unembedding layer reads off the argmax to produce `(a+b) mod p`.
This is a complete, mathematically verifiable description of the algorithm. Mechanistic
interpretability turned grokking from a mysterious empirical phenomenon into an understood one.

**Difference from saliency methods:**
Saliency methods identify *which inputs* the network is sensitive to, but not *how it processes*
those inputs. Mechanistic interpretability identifies the actual computational structure.

---

## 3. The Modular Arithmetic Task

**Task definition:**
Given two integers `a`, `b` drawn from `{0, 1, ..., p-1}`, predict `(a + b) mod p`.
The input sequence is the token triple `[a, b, =]` and the target is the token for the result.

**Dataset construction:**
All `p²` ordered pairs `(a, b)` are enumerated. For `p=113`, this yields `12,769` pairs.
30% (`≈ 3,831` pairs) are randomly assigned to the training set; 70% (`≈ 8,938`) are held out
as the test set. This split is *not* IID in the sensor usual sense: entire input pairs are
withheld, so the model must generalize to unseen `(a, b)` combinations.

**Why p must be prime:**
`Z/pZ` (integers mod p) forms a **field** when p is prime: it supports addition, subtraction,
multiplication, and division (multiplicative inverses exist for all nonzero elements).
This cyclic group structure enables discrete Fourier analysis (§5) and ensures that the
trigonometric circuit (§6) is well-defined. Composite p would make the group structure more
complex and break the clean Fourier decomposition.

**What a correct algorithmic solution looks like:**
The model must learn the identity `(a + b) mod p` using the trigonometric sum formula:
```
cos(ω(a+b)) = cos(ωa)cos(ωb) − sin(ωa)sin(ωb)
```
for specific frequencies `ω = 2πk/p`. This is a compact, weight-efficient representation that
generalizes to all `(a, b)` pairs — including those not in the training set.

---

## 4. The Transformer Architecture

**Architecture:** A 1-layer, 4-head transformer with **pre-LayerNorm** (LayerNorm applied
before each sublayer, not after).

**Hyperparameters (exact, from paper):**
- `d_model = 128` — embedding dimension
- `num_heads = 4` — attention heads (head_dim = 32)
- `d_mlp = 512` — MLP hidden dimension (= 4 × d_model)
- `activation = ReLU` (not GeLU — this is important for the Fourier structure)
- `num_layers = 1`
- `vocab_size = 114` — tokens 0..112 (operands) + token 113 (the `=` sign)
- `seq_len = 3` — three token positions: [a, b, =]

**Positional embeddings:** Learned (not sinusoidal), of shape `(3, d_model)`.

**Loss computation:** Cross-entropy loss is computed **only on the last position** (position 2,
the `=` sign), which predicts the result. The logits at positions 0 and 1 are not supervised.

**Why 1 layer is sufficient:**
The trig identity `cos(ω(a+b))` can be computed in a single pass:
1. Embed `a` and `b` into Fourier features.
2. Dot-product attention computes their outer products.
3. MLP nonlinearity (ReLU) combines them.
4. Unembedding reads off the answer.
No recurrence or depth is needed for this algorithm.

---

## 5. The Fourier Representation on Z/pZ

**Discrete Fourier analysis on a cyclic group:**
For the cyclic group `Z/pZ = {0, 1, ..., p-1}`, the Fourier basis consists of functions
`f_k(x) = cos(2πkx/p)` and `g_k(x) = sin(2πkx/p)` for `k = 0, 1, ..., (p-1)/2`.
These are the "frequency-k" basis functions — they oscillate k times around the group.

**What it means for a neural network to "use" Fourier features:**
The embedding matrix `W_E` maps token `x` to a vector in `R^{d_model}`. If the model learns
Fourier features, then the embedding of token `x` will have significant projections onto the
directions `cos(2πkx/p)` and `sin(2πkx/p)` for a small set of key frequencies `k`.
Concretely: two columns of `W_E` will approximately equal `[cos(2πkx/p)]_{x=0}^{p-1}` and
`[sin(2πkx/p)]_{x=0}^{p-1}` for the relevant k values.

**How Nanda measures Fourier structure (two progress measures from §8):**

*Fourier power spectrum:*
For each row `w_x` (the embedding of token `x`), compute its DFT; sum power across all x
at each frequency k. Key frequencies show high power; most others are near zero.

*Restricted loss:*
Project `W_E` onto the top-K Fourier components only (zeroing out all others), recompute
logits with this restricted embedding, compute cross-entropy loss. If this restricted loss
is low, the model has encoded the answer in its Fourier features — generalization is imminent.

*Fourier multiplicity:*
Count the number of frequencies k whose total Fourier power exceeds a threshold τ. A sparse
Fourier representation (few active frequencies) is the signature of the generalizing algorithm.

**Key frequencies:**
Only 3–5 frequencies (typically around `k = 14, 41, 42` for `p=113`) dominate the spectrum
of a fully grokked model. These are the frequencies that make the trig identity work.

---

## 6. The Trigonometric Circuit (Nanda's Mechanism)

**Core identity the model implements:**
```
cos(ω(a+b)) = cos(ωa)cos(ωb) − sin(ωa)sin(ωb)
sin(ω(a+b)) = sin(ωa)cos(ωb) + cos(ωa)sin(ωb)
```
for `ω = 2πk/p` at the key frequency k.

**Layer-by-layer breakdown:**

*Embedding layer:*
`W_E` maps each token `a` to a vector that lies in the span of
`{cos(ωa), sin(ωa)}` directions in `R^{d_model}`.
The learned positional embeddings `W_pos` modulate this by position.

*Attention layer:*
The attention mechanism computes dot products between Q and K vectors.
Because Q came from position 0 (`a`) and K from position 1 (`b`), the attention scores
encode bilinear products `cos(ωa)cos(ωb)` and `sin(ωa)sin(ωb)`, combining tokens a and b.
The "="-position values (V) aggregate these products and pass them to the MLP.

*MLP layer:*
The MLP with ReLU applies the nonlinear combination needed to isolate
`cos(ω(a+b)) = cos(ωa)cos(ωb) − sin(ωa)sin(ωb)` from the bilinear products.

*Unembedding layer:*
`W_U` maps the final hidden state to logits over the `p` output tokens.
The argmax of the logits gives `(a+b) mod p`.

**What logit lens reveals:**
Applying the unembedding matrix `W_U` after the embedding layer alone (before attention) shows
near-random predictions. After attention, the predictions improve. After the MLP, the model
reaches near-perfect accuracy. This confirms that the computation unfolds across the layers
in precisely the order described above.

---

## 7. Weight Decay and Why It Causes Grokking

**L2 regularization mechanics:**
Weight decay adds `λ/2 · ‖θ‖²` to the loss, penalizing parameter magnitudes.
At each step, AdamW applies: `θ ← θ − lr × (∇L + λθ)`.

**Why weight decay is necessary for grokking:**
In experiments without weight decay (`λ=0`), the model memorizes and stays memorized —
grokking never occurs. The Fourier Experiment §9 confirms this: grokking epoch diverges as
`λ → 0`.

**Nanda's efficiency hypothesis:**
The memorization solution (a lookup table) requires large, sparse weight matrices — it stores
one answer per training pair. This has high total weight magnitude `‖θ‖²`.
The generalizing trig circuit requires smaller, structured weights — it encodes a 3-frequency
algorithm. This has lower total weight magnitude.

The key insight: as training progresses, weight decay continuously shrinks all parameters.
The memorization solution becomes increasingly costly to maintain. At some point,
the cost of maintaining the memorization weights exceeds the benefit in training loss.
Gradient descent then reorganizes the weights to find the weight-efficient generalizing
circuit — this reorganization appears as the abrupt grokking transition.

**Connection to the three progress measures:**
The restricted loss begins to drop before test accuracy jumps (the Fourier features are
forming while the model still memorizes). The Fourier multiplicity decreases (only key
frequencies survive compression). The weight norm decreases (the lookup table weights shrink).
All three begin changing before test accuracy — they are leading indicators.

---

## 8. The Three Progress Measures

**Why train/test accuracy alone is insufficient:**
During the memorization phase, train accuracy = 100% and test accuracy ≈ chance. This is
stable for thousands of steps. Standard metrics give no warning of the impending generalization.
The three progress measures detect the internal reorganization while it is happening.

**Measure 1 — Restricted loss:**
*Definition:* Cross-entropy loss computed using only the top-K Fourier components of `W_E`
(project `W_E` onto the Fourier basis, keep top K by power, reconstruct, recompute logits).

*Intuition:* If Fourier structure is building in the embeddings, this restricted model can
already predict correctly — even if the full model still memorizes.

*PyTorch implementation:*
```python
def compute_restricted_loss(model, data, p, K=5):
    W_E = model.embedding.weight[:p]  # (p, d_model)
    # DFT over token indices
    freqs = torch.fft.rfft(W_E, dim=0)   # (p//2+1, d_model)
    power = freqs.abs() ** 2             # (p//2+1, d_model)
    top_k = power.sum(dim=1).topk(K).indices
    mask = torch.zeros(freqs.shape[0], device=W_E.device)
    mask[top_k] = 1.0
    W_E_restricted = torch.fft.irfft(freqs * mask.unsqueeze(1), n=p, dim=0)
    # ... swap embedding, compute loss, restore embedding
```

**Measure 2 — Fourier multiplicity:**
*Definition:* Number of frequencies k where the summed Fourier power of `W_E` exceeds
threshold τ (default τ = 0.01 × total power).

*Intuition:* Early in training, many frequencies have moderate power (spread-out, noise-like).
In the fully grokked model, only 3–5 frequencies dominate (the "key frequencies"). The number
of active frequencies decreases as the model transitions from memorization to generalization.

*PyTorch implementation:*
```python
def compute_fourier_multiplicity(W_E, p, tau_fraction=0.01):
    freqs = torch.fft.rfft(W_E[:p], dim=0)
    power = (freqs.abs() ** 2).sum(dim=1)  # (p//2+1,)
    threshold = tau_fraction * power.sum()
    return (power > threshold).sum().item()
```

**Measure 3 — Weight norm:**
*Definition:* `‖θ‖₂ = sqrt(Σ_i p_i.norm()²)` summed over all model parameters.

*Intuition:* Weight decay continuously shrinks `‖θ‖`. As the model drops the large-weight
memorization solution and adopts the compact Fourier circuit, the weight norm decreases faster.
This decrease precedes the test accuracy jump.

*PyTorch implementation:*
```python
def compute_weight_norm(model):
    return sum(p.norm() ** 2 for p in model.parameters()).sqrt().item()
```

All three measures begin changing **before** test accuracy jumps — they are leading indicators
of the impending grokking transition.

---

## 9. Further Explorations — Theoretical Motivation

**Weight decay sweep (λ ∈ {0.0, 0.1, 0.5, 1.0, 2.0}):**
- `λ=0`: grokking should not occur; the model memorizes indefinitely.
- `λ=0.1`: very slow compression; grokking may occur late or not at all in 20k steps.
- `λ=0.5`: moderate; grokking expected but slower than baseline.
- `λ=1.0`: baseline (paper setting); grokking expected around epoch 10k.
- `λ=2.0`: aggressive compression; grokking may occur faster, but risk of under-fitting the
  trig circuit (if weights are compressed before the algorithm forms).
- *Hypothesis:* grokking epoch ∝ 1/λ for λ in the useful range (0.1–2.0).

**Varying p (primes: 53, 97, 113, 127):**
- Larger p → larger group → more Fourier basis functions needed → a more complex circuit.
- The model needs more embedding dimensions to capture the necessary frequencies.
- *Hypothesis:* grokking epoch increases with p (roughly proportional to p or p²).
- Also tests *structural generality*: does the same trig circuit emerge for all prime p?

**Different operations (add, subtract, multiply mod p):**
- All are well-defined group operations on Z/pZ (or Z/pZ×).
- Addition and subtraction are equivalent by symmetry (subtract = add the negation).
- Multiplication: Z/pZ× is cyclic of order p−1. Every non-zero element is a power of a
  generator g. `a×b = g^{log_g(a) + log_g(b)}` — a different Fourier structure than addition.
- *Hypothesis:* All three operations grok, but at different epochs. Subtraction ≈ addition.
  Multiplication may require a different set of key frequencies.

**Model depth (1, 2, 3 layers):**
- The 1-layer model implements the trig identity in a single attention + MLP pass.
- 2-layer models have more representational power — they may find the circuit faster, or
  may learn unnecessary redundancy that delays grokking.
- 3-layer models may overfit the memorization solution longer (more parameters → longer to
  compress) or may find a qualitatively different algorithm entirely.
- *Hypothesis:* 2-layer grokks slightly faster than 1-layer; 3-layer grokks later. All reach
  similar final accuracy, but with different Fourier spectral profiles.
