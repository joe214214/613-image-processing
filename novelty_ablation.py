"""
Novelty Component Part 2: Ablation Study
Varies n_atoms in {64, 128, 256, 512} at sigma=25.
Saves results to ablation.json and a bar chart to ablation.png.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skimage.data import camera
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.linear_model import orthogonal_mp_gram
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d

PATCH_SIZE  = 8
C           = 1.15
N_TRAIN     = 50000
MAX_ITER    = 100
SIGMA_255   = 25
SIGMA       = SIGMA_255 / 255.0
ATOMS_LIST  = [64, 128, 256, 512]

image = camera().astype(np.float64) / 255.0
np.random.seed(42)
noisy = np.clip(image + np.random.normal(0, SIGMA, image.shape), 0, 1)
psnr_noisy = psnr(image, noisy)

results = {"noisy": round(psnr_noisy, 2), "atoms": {}}
print(f"sigma={SIGMA_255}, Noisy PSNR={psnr_noisy:.2f} dB\n")
print(f"{'n_atoms':>8} | {'PSNR':>6}")
print("-" * 18)

for n_atoms in ATOMS_LIST:
    print(f"  n_atoms={n_atoms} ...", end=" ", flush=True)

    patches    = extract_patches_2d(noisy, (PATCH_SIZE, PATCH_SIZE))
    n_patches  = patches.shape[0]
    patches_2d = patches.reshape(n_patches, -1).astype(np.float64)
    means      = patches_2d.mean(axis=1, keepdims=True)
    patches_2d -= means

    rng = np.random.default_rng(42)
    idx = rng.choice(n_patches, size=min(N_TRAIN, n_patches), replace=False)
    dico = MiniBatchDictionaryLearning(
        n_components=n_atoms, alpha=0.5, max_iter=MAX_ITER,
        batch_size=256, random_state=42, fit_algorithm='lars',
        max_no_improvement=None, verbose=0,
    )
    dico.fit(patches_2d[idx])
    D = dico.components_

    omp_tol  = C * (SIGMA ** 2) * PATCH_SIZE ** 2
    norms_sq = np.sum(patches_2d ** 2, axis=1)
    codes    = orthogonal_mp_gram(D @ D.T, D @ patches_2d.T,
                                  tol=omp_tol, norms_squared=norms_sq)
    rec  = (D.T @ codes).T + means
    rec3 = rec.reshape(-1, PATCH_SIZE, PATCH_SIZE)
    out  = np.clip(reconstruct_from_patches_2d(rec3, image.shape), 0, 1)
    p    = psnr(image, out)
    results["atoms"][n_atoms] = round(p, 2)
    print(f"{p:.2f} dB")

with open("ablation.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved ablation.json")

# ── Bar chart ────────────────────────────────────────────────
atoms  = list(results["atoms"].keys())
psnrs  = list(results["atoms"].values())

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar([str(a) for a in atoms], psnrs, color='steelblue', width=0.5)
ax.axhline(psnr_noisy, color='red', linestyle='--', label=f'Noisy ({psnr_noisy:.2f} dB)')
ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
ax.set_xlabel('Dictionary Size (n_atoms)')
ax.set_ylabel('PSNR (dB)')
ax.set_title(r'Ablation: Dictionary Size vs PSNR ($\sigma=25$)')
ax.legend()
ax.set_ylim(psnr_noisy - 1, max(psnrs) + 1.5)
plt.tight_layout()
plt.savefig('ablation.png', dpi=150)
print("Saved ablation.png")
