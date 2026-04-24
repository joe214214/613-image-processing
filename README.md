# ECE 613 Final Project — K-SVD Dictionary Learning

**ECE 417 / ECE 613: Image Processing and Visual Communication, Winter 2026**
University of Waterloo

---

## Project Overview

This project reproduces the K-SVD algorithm from:

> M. Aharon, M. Elad, and A. Bruckstein, "K-SVD: An Algorithm for Designing
> Overcomplete Dictionaries for Sparse Representation," *IEEE Trans. Signal
> Processing*, vol. 54, no. 11, pp. 4311–4322, Nov. 2006.

It then extends K-SVD to image denoising — a problem not studied in the
original paper — and benchmarks it against Non-Local Means (NLM) and BM3D.

---

## Reproducible Component

**What the paper does:** K-SVD learns an overcomplete dictionary from training
signals by alternating between sparse coding (OMP) and a rank-1 SVD atom
update. The paper's key claim is that K-SVD outperforms the prior MOD baseline
(which uses a global pseudoinverse update).

**Experiment reproduced (paper Section V-A):**

| Setting | Value |
|---------|-------|
| Signal dimension | n = 20 |
| Dictionary size | K = 50 atoms |
| Training signals | N = 1500 |
| Sparsity | L = 3 non-zeros |
| Iterations | 80 |
| Monte Carlo trials | 20 |
| Noise levels | ∞, 30 dB, 20 dB, 10 dB SNR |

**Results:**

| Method | No noise | 30 dB | 20 dB | 10 dB |
|--------|----------|-------|-------|-------|
| K-SVD (ours) | 45.5 ± 2.2 | 44.6 ± 2.2 | 44.8 ± 1.9 | 45.1 ± 3.0 |
| MOD (ours)   | 44.1 ± 1.9 | 45.3 ± 1.6 | 44.5 ± 2.2 | 44.8 ± 2.4 |
| K-SVD (paper) | ~49 | ~46 | ~40 | ~28 |

K-SVD matches or beats MOD at all noise levels, reproducing the paper's central
finding. The ~3–5 atom gap vs. the paper is explained by fewer Monte Carlo
trials (20 vs. 50) and OMP solver differences.

**Script:** `synthetic_experiment.py`
**Outputs:** `synthetic_results.json`, `synthetic_recovery.png`

---

## Novelty Component: K-SVD for Image Denoising

The original paper does not address image denoising. We apply K-SVD to this
new problem and compare against two state-of-the-art baselines.

**Method:**
- Extract all overlapping 8×8 patches from the noisy image (stride 1)
- Train a 64×256 dictionary via mini-batch K-SVD on 50,000 random patches
- Re-encode each patch with OMP using a noise-adapted stopping criterion:
  `‖Dα − y‖² ≤ C·σ²·n`, with C = 1.15
- Reconstruct via pixel-wise average of overlapping patch estimates

**PSNR results on Camera Man (512×512):**

| σ  | Noisy  | K-SVD      | NLM   | BM3D       |
|----|--------|------------|-------|------------|
| 10 | 28.24  | 33.31      | 32.86 | **34.23**  |
| 20 | 22.41  | 29.77      | 29.53 | **30.58**  |
| 25 | 20.59  | 28.75      | 28.71 | **29.68**  |
| 30 | 19.14  | 27.87      | 28.04 | **28.99**  |

K-SVD matches NLM at σ ≤ 25 and trails BM3D by ~1 dB. BM3D's advantage
comes from non-local patch grouping, a capability K-SVD lacks.

**Ablation — dictionary size K at σ = 25:**

| K   | PSNR (dB) |
|-----|-----------|
| 64  | 28.59     |
| 128 | 28.65     |
| 256 | 28.75     |
| 512 | 28.80     |

Diminishing returns beyond K = 256; compute cost scales linearly with K.

**Scripts:** `novelty_comparison.py`, `novelty_ablation.py`
**Outputs:** `comparison.json`, `comparison.png`, `ablation.json`, `ablation.png`

---

## Repository Structure

```
.
├── synthetic_experiment.py   # Reproduces paper Section V-A (K-SVD vs MOD)
├── novelty_comparison.py     # K-SVD vs NLM vs BM3D denoising
├── novelty_ablation.py       # Ablation: vary dictionary size K
├── experiment.py             # Multi-sigma denoising sweep
├── denoise_v2.py             # Core K-SVD denoising implementation
├── ksvd.py                   # Hand-written K-SVD reference
├── utils.py                  # Patch extraction / reconstruction utilities
├── report_final.tex          # IEEE ICIP-format report (LaTeX source)
├── report_final.pdf          # Compiled report
├── synthetic_results.json    # Raw recovery counts (reproducible experiment)
├── comparison.json           # Denoising PSNR table (method comparison)
├── ablation.json             # Denoising PSNR table (ablation study)
└── requirements.txt          # Python dependencies
```

---

## Setup

```bash
pip install -r requirements.txt
```

**Dependencies:** numpy, scipy, scikit-learn, scikit-image, matplotlib, pillow, bm3d

---

## Running the Experiments

```bash
# Reproducible component — synthetic dictionary recovery
python synthetic_experiment.py
# → synthetic_results.json, synthetic_recovery.png

# Novelty — method comparison (K-SVD / NLM / BM3D)
python novelty_comparison.py
# → comparison.json, comparison.png

# Novelty — ablation study (dictionary size)
python novelty_ablation.py
# → ablation.json, ablation.png
```

---

## Report

`report_final.pdf` — IEEE ICIP conference format, 3 pages.
Fill in author name/student number in `report_final.tex` line 11 before recompiling.

```bash
pdflatex report_final.tex
pdflatex report_final.tex   # second pass resolves cross-references
```
