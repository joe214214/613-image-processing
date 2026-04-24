"""
Visualizes the 256 dictionary atoms learned by K-SVD from Camera Man patches.
Saves a 16x16 grid of 8x8 atoms to dict_atoms.png.
"""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.data import camera
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d

PATCH_SIZE = 8
N_ATOMS    = 256
N_TRAIN    = 50000
SIGMA      = 25 / 255.0

rng = np.random.default_rng(42)

image = camera().astype(np.float64) / 255.0
np.random.seed(42)
noisy = np.clip(image + np.random.normal(0, SIGMA, image.shape), 0, 1)

patches    = extract_patches_2d(noisy, (PATCH_SIZE, PATCH_SIZE))
n_patches  = patches.shape[0]
patches_2d = patches.reshape(n_patches, -1)
means      = patches_2d.mean(axis=1, keepdims=True)
patches_2d = patches_2d - means

idx  = rng.choice(n_patches, size=min(N_TRAIN, n_patches), replace=False)
dico = MiniBatchDictionaryLearning(
    n_components=N_ATOMS, alpha=0.5, max_iter=100,
    batch_size=256, random_state=42, fit_algorithm='lars',
    max_no_improvement=None, verbose=0,
)
dico.fit(patches_2d[idx])
D = dico.components_   # shape (256, 64)

# Tile into 16x16 grid
grid_size = 16
atom_h = atom_w = PATCH_SIZE
border  = 1
canvas_h = grid_size * (atom_h + border) + border
canvas_w = grid_size * (atom_w + border) + border
canvas   = np.ones((canvas_h, canvas_w)) * 0.5

for i, atom in enumerate(D):
    row = i // grid_size
    col = i %  grid_size
    patch = atom.reshape(atom_h, atom_w)
    vmax  = np.abs(patch).max()
    if vmax > 0:
        patch = patch / vmax          # normalise to [-1, 1]
    patch = (patch + 1) / 2          # shift to [0, 1] for display
    r0 = border + row * (atom_h + border)
    c0 = border + col * (atom_w + border)
    canvas[r0:r0+atom_h, c0:c0+atom_w] = patch

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(canvas, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
ax.set_title('256 Learned Dictionary Atoms (K-SVD, Camera Man, $\\sigma=25$)',
             fontsize=10)
ax.axis('off')
plt.tight_layout()
plt.savefig('dict_atoms.png', dpi=150)
print("Saved dict_atoms.png")
