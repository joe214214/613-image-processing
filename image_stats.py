"""
Quantitative comparison of Camera Man vs Astronaut for the denoising analysis.
Computes metrics that explain why NLM+K-SVD helps on one image but not the other:

  - Mean gradient magnitude (overall "texturedness")
  - High-frequency energy ratio (fraction of DCT energy above cutoff)
  - Patch self-similarity (NLM's core assumption): for each of 2000 random
    8x8 patches, the mean Euclidean distance to its 10 nearest neighbours
    among 5000 other random patches
"""
import json
import numpy as np
from scipy import fftpack
from skimage.data import camera, astronaut
from skimage.color import rgb2gray
from sklearn.feature_extraction.image import extract_patches_2d

PATCH = 8
N_QUERY = 2000
N_REF   = 5000
K_NN    = 10
rng = np.random.default_rng(0)

def grad_mag(img):
    gy, gx = np.gradient(img)
    return float(np.mean(np.sqrt(gy ** 2 + gx ** 2)))

def hf_energy_ratio(img, cutoff_frac=0.25):
    F = np.abs(fftpack.dct(fftpack.dct(img, axis=0, norm='ortho'),
                           axis=1, norm='ortho'))
    h, w = img.shape
    cy, cx = int(h * cutoff_frac), int(w * cutoff_frac)
    lf = (F[:cy, :cx] ** 2).sum()
    total = (F ** 2).sum()
    return float((total - lf) / total)

def patch_self_similarity(img):
    patches = extract_patches_2d(img, (PATCH, PATCH)).reshape(-1, PATCH * PATCH)
    # mean-subtract each patch (same as the denoising pipeline)
    patches = patches - patches.mean(axis=1, keepdims=True)
    n = patches.shape[0]
    q_idx = rng.choice(n, N_QUERY, replace=False)
    r_idx = rng.choice(n, N_REF,   replace=False)
    Q = patches[q_idx]
    R = patches[r_idx]
    # Euclidean distance from every Q to every R
    d2 = (Q ** 2).sum(1, keepdims=True) + (R ** 2).sum(1) - 2 * Q @ R.T
    d2 = np.clip(d2, 0, None)
    # mean distance to k-nearest neighbours (smaller = more redundant)
    part = np.partition(d2, K_NN, axis=1)[:, :K_NN]
    return float(np.mean(np.sqrt(part)))

imgs = {
    "CameraMan": camera().astype(np.float64) / 255.0,
    "Astronaut": rgb2gray(astronaut()).astype(np.float64),
}

results = {}
for name, img in imgs.items():
    g = grad_mag(img)
    h = hf_energy_ratio(img)
    s = patch_self_similarity(img)
    results[name] = {"grad_mag": round(g, 4),
                     "hf_ratio": round(h, 4),
                     "knn_dist": round(s, 4)}
    print(f"{name:>10}  grad={g:.4f}  HF-energy={h:.4f}  kNN-dist={s:.4f}")

with open("image_stats.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved image_stats.json")
