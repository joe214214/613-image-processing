"""
Regenerates comparison figures without BM3D.
Produces: comparison_nobm3d.png and nlm_ksvd_final.png
"""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.data import camera
from skimage.restoration import denoise_nl_means
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.linear_model import orthogonal_mp_gram
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d

PATCH_SIZE = 8; N_ATOMS = 256; C = 1.15; N_TRAIN = 50000; MAX_ITER = 100

image = camera().astype(np.float64) / 255.0
sigma_255 = 25; sigma = sigma_255 / 255.0
np.random.seed(42)
noisy = np.clip(image + np.random.normal(0, sigma, image.shape), 0, 1)

def ksvd_denoise(noisy, sigma):
    patches = extract_patches_2d(noisy, (PATCH_SIZE, PATCH_SIZE))
    n = patches.shape[0]
    p2d = patches.reshape(n, -1).astype(np.float64)
    means = p2d.mean(axis=1, keepdims=True); p2d -= means
    idx = np.random.default_rng(42).choice(n, size=min(N_TRAIN, n), replace=False)
    dico = MiniBatchDictionaryLearning(n_components=N_ATOMS, alpha=0.5, max_iter=MAX_ITER,
               batch_size=256, random_state=42, fit_algorithm='lars',
               max_no_improvement=None, verbose=0)
    dico.fit(p2d[idx]); D = dico.components_
    omp_tol = C * sigma**2 * PATCH_SIZE**2
    codes = orthogonal_mp_gram(D @ D.T, D @ p2d.T, tol=omp_tol,
                               norms_squared=np.sum(p2d**2, axis=1))
    rec = (D.T @ codes).T + means
    return np.clip(reconstruct_from_patches_2d(rec.reshape(-1, PATCH_SIZE, PATCH_SIZE),
                                               noisy.shape), 0, 1)

def nlm_denoise(noisy, sigma):
    return denoise_nl_means(noisy, h=0.8*sigma, patch_size=7,
                            patch_distance=11, fast_mode=True)

def nlm_ksvd_denoise(noisy, sigma):
    nlm_out = nlm_denoise(noisy, sigma)
    patches_nlm = extract_patches_2d(nlm_out, (PATCH_SIZE, PATCH_SIZE))
    n = patches_nlm.shape[0]
    p2d_nlm = patches_nlm.reshape(n, -1).astype(np.float64)
    means_nlm = p2d_nlm.mean(axis=1, keepdims=True); p2d_nlm -= means_nlm
    idx = np.random.default_rng(42).choice(n, size=min(N_TRAIN, n), replace=False)
    dico = MiniBatchDictionaryLearning(n_components=N_ATOMS, alpha=0.5, max_iter=MAX_ITER,
               batch_size=256, random_state=42, fit_algorithm='lars',
               max_no_improvement=None, verbose=0)
    dico.fit(p2d_nlm[idx]); D = dico.components_
    patches_noisy = extract_patches_2d(noisy, (PATCH_SIZE, PATCH_SIZE))
    p2d_noisy = patches_noisy.reshape(n, -1).astype(np.float64)
    means_noisy = p2d_noisy.mean(axis=1, keepdims=True); p2d_noisy -= means_noisy
    omp_tol = C * sigma**2 * PATCH_SIZE**2
    codes = orthogonal_mp_gram(D @ D.T, D @ p2d_noisy.T, tol=omp_tol,
                               norms_squared=np.sum(p2d_noisy**2, axis=1))
    rec = (D.T @ codes).T + means_noisy
    return np.clip(reconstruct_from_patches_2d(rec.reshape(-1, PATCH_SIZE, PATCH_SIZE),
                                               noisy.shape), 0, 1)

from skimage.metrics import peak_signal_noise_ratio as psnr

print("Computing K-SVD..."); ksvd_out = ksvd_denoise(noisy, sigma)
print("Computing NLM...");   nlm_out  = nlm_denoise(noisy, sigma)
print("Computing NLM+K-SVD..."); combo_out = nlm_ksvd_denoise(noisy, sigma)

pk = round(psnr(image, ksvd_out), 2)
pn = round(psnr(image, nlm_out),  2)
pc = round(psnr(image, combo_out),2)
pnoisy = round(psnr(image, noisy), 2)
print(f"K-SVD={pk} NLM={pn} NLM+K-SVD={pc}")

# Figure 1: K-SVD vs NLM comparison (4 panels, no BM3D)
panels4 = [
    ("Original",               image),
    (f"Noisy\n({pnoisy:.2f} dB)", noisy),
    (f"K-SVD\n({pk:.2f} dB)",  ksvd_out),
    (f"NLM\n({pn:.2f} dB)",    nlm_out),
]
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
for ax, (title, im) in zip(axes, panels4):
    ax.imshow(im, cmap='gray', vmin=0, vmax=1)
    ax.set_title(title, fontsize=10); ax.axis('off')
plt.suptitle(r'K-SVD vs NLM on Camera Man ($\sigma=25$)', fontsize=11)
plt.tight_layout()
plt.savefig('comparison_nobm3d.png', dpi=150)
print("Saved comparison_nobm3d.png")

# Figure 2: NLM+K-SVD comparison (5 panels, no BM3D)
panels5 = [
    ("Original",               image),
    (f"Noisy\n({pnoisy:.2f} dB)", noisy),
    (f"K-SVD\n({pk:.2f} dB)",  ksvd_out),
    (f"NLM\n({pn:.2f} dB)",    nlm_out),
    (f"NLM+K-SVD\n({pc:.2f} dB)", combo_out),
]
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
for ax, (title, im) in zip(axes, panels5):
    ax.imshow(im, cmap='gray', vmin=0, vmax=1)
    ax.set_title(title, fontsize=10); ax.axis('off')
plt.suptitle(r'NLM+K-SVD vs Baselines on Camera Man ($\sigma=25$)', fontsize=11)
plt.tight_layout()
plt.savefig('nlm_ksvd_final.png', dpi=150)
print("Saved nlm_ksvd_final.png")
