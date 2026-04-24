import numpy as np
from skimage.data import camera
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import extract_patches, reconstruct_image
from ksvd import ksvd, omp

image = camera().astype(np.float64) / 255.0

np.random.seed(42)
sigma = 0.1
noisy = image + np.random.normal(0, sigma, image.shape)
noisy = np.clip(noisy, 0, 1)

print(f"Image shape: {image.shape}")
print(f"PSNR (noisy): {psnr(image, noisy):.2f} dB")

patch_size = 8
print("Extracting patches...")
Y, positions = extract_patches(noisy, patch_size=patch_size, step=4)
print(f"Total patches: {Y.shape[1]}")

means = Y.mean(axis=0, keepdims=True)
Y -= means

print("Running K-SVD...")
D, X = ksvd(Y, n_atoms=128, n_nonzero=6, n_iter=10)

print("Reconstructing...")
Y_denoised = D @ X
Y_denoised += means

denoised = reconstruct_image(Y_denoised, positions, image.shape, patch_size)
denoised = np.clip(denoised, 0, 1)

print(f"PSNR (denoised): {psnr(image, denoised):.2f} dB")
print(f"PSNR gain:       {psnr(image, denoised) - psnr(image, noisy):.2f} dB")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(image, cmap='gray'); axes[0].set_title('Original'); axes[0].axis('off')
axes[1].imshow(noisy, cmap='gray'); axes[1].set_title(f'Noisy ({psnr(image, noisy):.1f} dB)'); axes[1].axis('off')
axes[2].imshow(denoised, cmap='gray'); axes[2].set_title(f'Denoised ({psnr(image, denoised):.1f} dB)'); axes[2].axis('off')
plt.tight_layout()
plt.savefig('result.png', dpi=150)
print("Saved to result.png")
