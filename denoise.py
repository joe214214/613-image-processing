import numpy as np
from skimage import color
from skimage.data import camera  # 内置测试图，不需要外部图片
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib
matplotlib.use('Agg')  # 无显示器环境
import matplotlib.pyplot as plt

from utils import extract_patches, reconstruct_image
from ksvd import ksvd, omp

# 1. 用内置测试图（512x512 灰度）
image = camera().astype(np.float64) / 255.0

# 2. 加高斯噪声
np.random.seed(42)
sigma = 0.1
noisy = image + np.random.normal(0, sigma, image.shape)
noisy = np.clip(noisy, 0, 1)

print(f"原图尺寸: {image.shape}")
print(f"PSNR（带噪）: {psnr(image, noisy):.2f} dB")

# 3. 切 patch（用 step=4 加速，正式实验用 step=1）
patch_size = 8
print("正在切 patch...")
Y, positions = extract_patches(noisy, patch_size=patch_size, step=4)
print(f"共 {Y.shape[1]} 个 patch")

# 4. 减均值
means = Y.mean(axis=0, keepdims=True)
Y -= means

# 5. K-SVD 学字典
print("开始 K-SVD 训练...")
D, X = ksvd(Y, n_atoms=128, n_nonzero=6, n_iter=10)

# 6. 稀疏重建
print("重建图像...")
Y_denoised = D @ X
Y_denoised += means

# 7. 拼回图像
denoised = reconstruct_image(Y_denoised, positions, image.shape, patch_size)
denoised = np.clip(denoised, 0, 1)

# 8. 结果
print(f"PSNR（去噪）: {psnr(image, denoised):.2f} dB")
print(f"PSNR 提升:    {psnr(image, denoised) - psnr(image, noisy):.2f} dB")

# 9. 保存图片
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(image, cmap='gray'); axes[0].set_title('原图')
axes[0].axis('off')
axes[1].imshow(noisy, cmap='gray'); axes[1].set_title(f'带噪 ({psnr(image, noisy):.1f} dB)')
axes[1].axis('off')
axes[2].imshow(denoised, cmap='gray'); axes[2].set_title(f'去噪 ({psnr(image, denoised):.1f} dB)')
axes[2].axis('off')
plt.tight_layout()
plt.savefig('result.png', dpi=150)
print("结果已保存到 result.png")
