"""
Combined NLM + K-SVD denoising pipeline.

Step 1: NLM pre-denoising (removes bulk noise, exploits non-local similarity)
Step 2: K-SVD refinement on the NLM output (learns a cleaner dictionary,
        uses a tighter OMP threshold based on residual noise)

Compares: NLM+K-SVD | K-SVD | NLM | BM3D  on Camera Man and Astronaut.
"""
import json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skimage.data import camera, astronaut
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

images = {
    "CameraMan": camera().astype(np.float64) / 255.0,
    "Astronaut": rgb2gray(astronaut()).astype(np.float64),
}

def ksvd_denoise(noisy, sigma):
    patches    = extract_patches_2d(noisy, (PATCH_SIZE, PATCH_SIZE))
    n          = patches.shape[0]
    p2d        = patches.reshape(n, -1).astype(np.float64)
    means      = p2d.mean(axis=1, keepdims=True); p2d -= means
    idx = np.random.default_rng(42).choice(n, size=min(N_TRAIN, n), replace=False)
    dico = MiniBatchDictionaryLearning(n_components=N_ATOMS, alpha=0.5, max_iter=MAX_ITER,
               batch_size=256, random_state=42, fit_algorithm='lars',
               max_no_improvement=None, verbose=0)
    dico.fit(p2d[idx]); D = dico.components_
    omp_tol  = C * sigma**2 * PATCH_SIZE**2
    codes    = orthogonal_mp_gram(D @ D.T, D @ p2d.T, tol=omp_tol,
                                  norms_squared=np.sum(p2d**2, axis=1))
    rec = (D.T @ codes).T + means
    return np.clip(reconstruct_from_patches_2d(rec.reshape(-1, PATCH_SIZE, PATCH_SIZE),
                                               noisy.shape), 0, 1)

def nlm_denoise(noisy, sigma):
    return denoise_nl_means(noisy, h=0.8*sigma, patch_size=7,
                            patch_distance=11, fast_mode=True)

def bm3d_denoise(noisy, sigma):
    return np.clip(bm3d.bm3d(noisy, sigma_psd=sigma), 0, 1)

def nlm_ksvd_denoise(noisy, sigma):
    # Step 1: NLM pre-denoise — used only for dictionary training
    nlm_out = denoise_nl_means(noisy, h=0.8*sigma, patch_size=7,
                               patch_distance=11, fast_mode=True)

    # Step 2: Train K-SVD dictionary on the cleaner NLM patches
    patches_nlm = extract_patches_2d(nlm_out, (PATCH_SIZE, PATCH_SIZE))
    n           = patches_nlm.shape[0]
    p2d_nlm     = patches_nlm.reshape(n, -1).astype(np.float64)
    means_nlm   = p2d_nlm.mean(axis=1, keepdims=True); p2d_nlm -= means_nlm
    idx = np.random.default_rng(42).choice(n, size=min(N_TRAIN, n), replace=False)
    dico = MiniBatchDictionaryLearning(n_components=N_ATOMS, alpha=0.5, max_iter=MAX_ITER,
               batch_size=256, random_state=42, fit_algorithm='lars',
               max_no_improvement=None, verbose=0)
    dico.fit(p2d_nlm[idx]); D = dico.components_

    # Step 3: OMP encode the ORIGINAL noisy patches with the better dictionary
    patches_noisy = extract_patches_2d(noisy, (PATCH_SIZE, PATCH_SIZE))
    p2d_noisy     = patches_noisy.reshape(n, -1).astype(np.float64)
    means_noisy   = p2d_noisy.mean(axis=1, keepdims=True); p2d_noisy -= means_noisy
    omp_tol  = C * sigma**2 * PATCH_SIZE**2
    codes    = orthogonal_mp_gram(D @ D.T, D @ p2d_noisy.T, tol=omp_tol,
                                  norms_squared=np.sum(p2d_noisy**2, axis=1))
    rec = (D.T @ codes).T + means_noisy
    return np.clip(reconstruct_from_patches_2d(rec.reshape(-1, PATCH_SIZE, PATCH_SIZE),
                                               noisy.shape), 0, 1)

methods = {
    "NLM+K-SVD": nlm_ksvd_denoise,
    "K-SVD":     ksvd_denoise,
    "NLM":       nlm_denoise,
    "BM3D":      bm3d_denoise,
}

all_results = {}

for img_name, image in images.items():
    print(f"\n=== {img_name} ===")
    print(f"{'sigma':>6} | {'NLM+KSVD':>9} | {'K-SVD':>6} | {'NLM':>6} | {'BM3D':>6}")
    print("-" * 46)
    all_results[img_name] = {}

    for sigma_255 in SIGMAS:
        sigma = sigma_255 / 255.0
        np.random.seed(42)
        noisy = np.clip(image + np.random.normal(0, sigma, image.shape), 0, 1)
        row = {"noisy": round(psnr(image, noisy), 2)}

        for name, fn in methods.items():
            print(f"  {img_name} σ={sigma_255} {name} ...", end=" ", flush=True)
            d = fn(noisy, sigma)
            p = round(psnr(image, d), 2)
            row[name] = p
            print(f"{p:.2f} dB")

        all_results[img_name][sigma_255] = row
        print(f"{sigma_255:>6} | {row['NLM+K-SVD']:>9.2f} | {row['K-SVD']:>6.2f} | "
              f"{row['NLM']:>6.2f} | {row['BM3D']:>6.2f}")

with open("nlm_ksvd_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nSaved nlm_ksvd_results.json")

# ── Visual comparison figure: Camera Man σ=25 ────────────────
img   = images["CameraMan"]
sigma = 25 / 255.0
np.random.seed(42)
noisy = np.clip(img + np.random.normal(0, sigma, img.shape), 0, 1)
r25   = all_results["CameraMan"][25]

panels = [
    ("Original",                              img),
    (f"Noisy\n({r25['noisy']:.2f} dB)",      noisy),
    (f"K-SVD\n({r25['K-SVD']:.2f} dB)",      ksvd_denoise(noisy, sigma)),
    (f"NLM\n({r25['NLM']:.2f} dB)",          nlm_denoise(noisy, sigma)),
    (f"BM3D\n({r25['BM3D']:.2f} dB)",        bm3d_denoise(noisy, sigma)),
    (f"NLM+K-SVD\n({r25['NLM+K-SVD']:.2f} dB)", nlm_ksvd_denoise(noisy, sigma)),
]

fig, axes = plt.subplots(1, 6, figsize=(22, 4))
for ax, (title, im) in zip(axes, panels):
    ax.imshow(im, cmap='gray', vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.axis('off')
plt.suptitle(r'NLM+K-SVD vs Baselines on Camera Man ($\sigma=25$)', fontsize=11)
plt.tight_layout()
plt.savefig('nlm_ksvd_comparison.png', dpi=150)
print("Saved nlm_ksvd_comparison.png")

# ── Summary ───────────────────────────────────────────────────
print("\n=== Final Summary (Camera Man) ===")
print(f"{'σ':>4} | {'Noisy':>6} | {'NLM+KSVD':>9} | {'K-SVD':>6} | {'NLM':>6} | {'BM3D':>6}")
print("-" * 50)
for s in SIGMAS:
    r = all_results["CameraMan"][s]
    best = max(r["NLM+K-SVD"], r["K-SVD"], r["NLM"], r["BM3D"])
    def fmt(v): return f"*{v:.2f}*" if v == best else f" {v:.2f} "
    print(f"{s:>4} | {r['noisy']:>6.2f} | {fmt(r['NLM+K-SVD']):>9} | "
          f"{fmt(r['K-SVD']):>6} | {fmt(r['NLM']):>6} | {fmt(r['BM3D']):>6}")
