import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.linear_model import orthogonal_mp_gram
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
from skimage.data import camera
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Parameters ──────────────────────────────────────────────
patch_size  = 8
n_atoms     = 256
sigma       = 0.1       # noise std (image range [0,1])
C           = 1.15      # noise tolerance factor (from paper)
n_train     = 50000     # patches used for dictionary training
max_iter    = 100       # dictionary learning iterations
n_nonzero   = 6        # OMP sparsity (fallback if tol unused)
# ─────────────────────────────────────────────────────────────

# 1. Load image and add noise
print("=" * 50)
image = camera().astype(np.float64) / 255.0
np.random.seed(42)
noisy = np.clip(image + np.random.normal(0, sigma, image.shape), 0, 1)
print(f"Image shape      : {image.shape}")
print(f"PSNR (noisy)     : {psnr(image, noisy):.2f} dB")

# 2. Extract ALL patches with step=1 (key difference from v1)
print("\n[1/4] Extracting patches (step=1)...")
patches     = extract_patches_2d(noisy, (patch_size, patch_size))
n_patches   = patches.shape[0]
patches_2d  = patches.reshape(n_patches, -1).astype(np.float64)
patch_means = patches_2d.mean(axis=1, keepdims=True)
patches_2d -= patch_means
print(f"Total patches    : {n_patches}")

# 3. Train dictionary on a random subset
print(f"\n[2/4] Training dictionary ({n_atoms} atoms, {max_iter} iters)...")
rng       = np.random.default_rng(42)
train_idx = rng.choice(n_patches, size=min(n_train, n_patches), replace=False)
dico = MiniBatchDictionaryLearning(
    n_components=n_atoms,
    alpha=0.5,
    max_iter=max_iter,
    batch_size=256,
    random_state=42,
    fit_algorithm='lars',
    verbose=1,
)
dico.fit(patches_2d[train_idx])
D = dico.components_          # shape: (n_atoms, patch_dim)
print("Dictionary training done.")

# 4. OMP sparse coding for ALL patches
#    Stopping criterion: ||residual||^2 <= C * sigma^2 * patch_dim
print("\n[3/4] Sparse coding all patches with OMP...")
patch_dim    = patch_size ** 2
omp_tol      = C * (sigma ** 2) * patch_dim  # per-patch squared residual threshold
norms_sq     = np.sum(patches_2d ** 2, axis=1)  # ||y_i||^2 for each patch

Gram = D @ D.T                  # (n_atoms, n_atoms)
Xy   = (D @ patches_2d.T)       # (n_atoms, n_patches)

codes = orthogonal_mp_gram(
    Gram, Xy,
    tol=omp_tol,
    norms_squared=norms_sq,
)                               # (n_atoms, n_patches)

# 5. Reconstruct patches and average overlaps
print("\n[4/4] Reconstructing image...")
patches_recon    = (D.T @ codes).T   # (n_patches, patch_dim)
patches_recon   += patch_means
patches_recon_3d = patches_recon.reshape(-1, patch_size, patch_size)
denoised         = reconstruct_from_patches_2d(patches_recon_3d, image.shape)
denoised         = np.clip(denoised, 0, 1)

# 6. Results
psnr_noisy    = psnr(image, noisy)
psnr_denoised = psnr(image, denoised)
print("\n" + "=" * 50)
print(f"PSNR (noisy)     : {psnr_noisy:.2f} dB")
print(f"PSNR (denoised)  : {psnr_denoised:.2f} dB")
print(f"PSNR gain        : {psnr_denoised - psnr_noisy:.2f} dB")
print("=" * 50)

# 7. Save comparison figure
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].imshow(image,   cmap='gray'); axes[0].set_title('Original');                          axes[0].axis('off')
axes[1].imshow(noisy,   cmap='gray'); axes[1].set_title(f'Noisy ({psnr_noisy:.1f} dB)');     axes[1].axis('off')
axes[2].imshow(denoised,cmap='gray'); axes[2].set_title(f'Denoised ({psnr_denoised:.1f} dB)');axes[2].axis('off')
plt.suptitle('K-SVD Image Denoising (v2)', fontsize=13)
plt.tight_layout()
plt.savefig('result_v2.png', dpi=150)
print("Saved to result_v2.png")
