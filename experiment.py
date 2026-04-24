"""
Reproduces Table 1 from Elad & Aharon 2006 across multiple noise levels.
Tests sigma = 10, 20, 25, 30 (on 0-255 scale) on the standard Camera Man image.
Saves results to results.json for use in the report.
"""
import json
import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.linear_model import orthogonal_mp_gram
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
from skimage.data import camera
from skimage.metrics import peak_signal_noise_ratio as psnr

# Paper reference values (Elad & Aharon 2006, Camera Man / similar images)
PAPER_PSNR = {10: 35.90, 20: 31.36, 25: 30.02, 30: 29.03}

PATCH_SIZE  = 8
N_ATOMS     = 256
C           = 1.15
N_TRAIN     = 50000
MAX_ITER    = 100

image_255 = camera()                              # uint8, 0-255
image     = image_255.astype(np.float64) / 255.0  # float, 0-1


def denoise(sigma_255):
    sigma = sigma_255 / 255.0
    np.random.seed(42)
    noisy = np.clip(image + np.random.normal(0, sigma, image.shape), 0, 1)

    # Extract patches
    patches    = extract_patches_2d(noisy, (PATCH_SIZE, PATCH_SIZE))
    n_patches  = patches.shape[0]
    patches_2d = patches.reshape(n_patches, -1).astype(np.float64)
    means      = patches_2d.mean(axis=1, keepdims=True)
    patches_2d -= means

    # Train dictionary
    rng       = np.random.default_rng(42)
    idx       = rng.choice(n_patches, size=min(N_TRAIN, n_patches), replace=False)
    dico = MiniBatchDictionaryLearning(
        n_components=N_ATOMS, alpha=0.5, max_iter=MAX_ITER,
        batch_size=256, random_state=42, fit_algorithm='lars',
        max_no_improvement=None, verbose=0,
    )
    dico.fit(patches_2d[idx])
    D = dico.components_

    # OMP with noise threshold
    omp_tol  = C * (sigma ** 2) * PATCH_SIZE ** 2
    norms_sq = np.sum(patches_2d ** 2, axis=1)
    Gram     = D @ D.T
    Xy       = D @ patches_2d.T
    codes    = orthogonal_mp_gram(Gram, Xy, tol=omp_tol, norms_squared=norms_sq)

    # Reconstruct
    rec  = (D.T @ codes).T + means
    rec3 = rec.reshape(-1, PATCH_SIZE, PATCH_SIZE)
    out  = np.clip(reconstruct_from_patches_2d(rec3, image.shape), 0, 1)

    return psnr(image, noisy), psnr(image, out)


results = {}
print(f"{'sigma':>6} | {'PSNR noisy':>10} | {'PSNR ours':>10} | {'PSNR paper':>11} | {'gap':>6}")
print("-" * 55)
for sigma_255 in [10, 20, 25, 30]:
    print(f"  sigma={sigma_255:2d} ... ", end="", flush=True)
    psnr_noisy, psnr_ours = denoise(sigma_255)
    psnr_paper = PAPER_PSNR[sigma_255]
    gap = psnr_ours - psnr_paper
    results[sigma_255] = {
        "psnr_noisy": round(psnr_noisy, 2),
        "psnr_ours":  round(psnr_ours,  2),
        "psnr_paper": psnr_paper,
        "gap":        round(gap, 2),
    }
    print(f"{sigma_255:>6} | {psnr_noisy:>10.2f} | {psnr_ours:>10.2f} | {psnr_paper:>11.2f} | {gap:>+6.2f}")

with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to results.json")
