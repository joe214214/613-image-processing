"""
Novelty Component: denoising on a second test image (Astronaut, treated as 'Lena').
Uses the same K-SVD / NLM / BM3D pipeline as novelty_comparison.py.
Saves lena_results.json and lena_comparison.png.
"""
import json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.restoration import denoise_nl_means
import bm3d

from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.linear_model import orthogonal_mp_gram
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d

PATCH_SIZE = 8
N_ATOMS    = 256
C          = 1.15
N_TRAIN    = 50000
MAX_ITER   = 100
SIGMAS     = [10, 20, 25, 30]

image = rgb2gray(astronaut()).astype(np.float64)  # already in [0, 1]

def ksvd_denoise(noisy, sigma):
    patches    = extract_patches_2d(noisy, (PATCH_SIZE, PATCH_SIZE))
    n_patches  = patches.shape[0]
    patches_2d = patches.reshape(n_patches, -1).astype(np.float64)
    means      = patches_2d.mean(axis=1, keepdims=True)
    patches_2d -= means

    rng = np.random.default_rng(42)
    idx = rng.choice(n_patches, size=min(N_TRAIN, n_patches), replace=False)
    dico = MiniBatchDictionaryLearning(
        n_components=N_ATOMS, alpha=0.5, max_iter=MAX_ITER,
        batch_size=256, random_state=42, fit_algorithm='lars',
        max_no_improvement=None, verbose=0,
    )
    dico.fit(patches_2d[idx])
    D = dico.components_

    omp_tol  = C * (sigma ** 2) * PATCH_SIZE ** 2
    norms_sq = np.sum(patches_2d ** 2, axis=1)
    codes    = orthogonal_mp_gram(D @ D.T, D @ patches_2d.T,
                                  tol=omp_tol, norms_squared=norms_sq)
    rec  = (D.T @ codes).T + means
    rec3 = rec.reshape(-1, PATCH_SIZE, PATCH_SIZE)
    return np.clip(reconstruct_from_patches_2d(rec3, image.shape), 0, 1)

def nlm_denoise(noisy, sigma):
    return denoise_nl_means(noisy, h=0.8 * sigma, patch_size=7,
                             patch_distance=11, fast_mode=True)

def bm3d_denoise(noisy, sigma):
    return np.clip(bm3d.bm3d(noisy, sigma_psd=sigma), 0, 1)

methods = {"K-SVD": ksvd_denoise, "NLM": nlm_denoise, "BM3D": bm3d_denoise}
results = {}

print(f"{'sigma':>6} | {'Method':>8} | {'PSNR':>6}")
print("-" * 28)

for sigma_255 in SIGMAS:
    sigma = sigma_255 / 255.0
    np.random.seed(42)
    noisy = np.clip(image + np.random.normal(0, sigma, image.shape), 0, 1)
    results[sigma_255] = {"noisy": round(psnr(image, noisy), 2)}

    for name, fn in methods.items():
        print(f"  sigma={sigma_255}, {name} ...", end=" ", flush=True)
        denoised = fn(noisy, sigma)
        p = psnr(image, denoised)
        results[sigma_255][name] = round(p, 2)
        print(f"{p:.2f} dB")

with open("lena_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved lena_results.json")

# Visual comparison at sigma=25
sigma_255 = 25
sigma = sigma_255 / 255.0
np.random.seed(42)
noisy = np.clip(image + np.random.normal(0, sigma, image.shape), 0, 1)

panels = {
    "Original":  image,
    f"Noisy\n({results[25]['noisy']:.1f} dB)":  noisy,
    f"K-SVD\n({results[25]['K-SVD']:.1f} dB)":  ksvd_denoise(noisy, sigma),
    f"NLM\n({results[25]['NLM']:.1f} dB)":       nlm_denoise(noisy, sigma),
    f"BM3D\n({results[25]['BM3D']:.1f} dB)":     bm3d_denoise(noisy, sigma),
}

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for ax, (title, img) in zip(axes, panels.items()):
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.axis('off')
plt.suptitle(r'Method Comparison on Astronaut at $\sigma=25$', fontsize=12)
plt.tight_layout()
plt.savefig('lena_comparison.png', dpi=150)
print("Saved lena_comparison.png")

print("\nSummary:")
print(f"{'sigma':>6} | {'Noisy':>6} | {'K-SVD':>6} | {'NLM':>6} | {'BM3D':>6}")
print("-" * 42)
for s in SIGMAS:
    r = results[s]
    print(f"{s:>6} | {r['noisy']:>6.2f} | {r['K-SVD']:>6.2f} | {r['NLM']:>6.2f} | {r['BM3D']:>6.2f}")
