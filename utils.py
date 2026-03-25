import numpy as np


def extract_patches(image, patch_size=8, step=1):
    h, w = image.shape
    patches = []
    positions = []
    for i in range(0, h - patch_size + 1, step):
        for j in range(0, w - patch_size + 1, step):
            patch = image[i:i+patch_size, j:j+patch_size].flatten()
            patches.append(patch)
            positions.append((i, j))
    return np.array(patches).T, positions  # (patch_dim, n_patches)


def reconstruct_image(patches, positions, image_shape, patch_size=8):
    h, w = image_shape
    image = np.zeros((h, w))
    count = np.zeros((h, w))
    for idx, (i, j) in enumerate(positions):
        patch = patches[:, idx].reshape(patch_size, patch_size)
        image[i:i+patch_size, j:j+patch_size] += patch
        count[i:i+patch_size, j:j+patch_size] += 1
    return image / count
