from ivclab.utils import imread, calc_psnr
from ivclab.quantization.quantizers import uniquant, inv_uniquant, lloyd_max, inv_lloyd_max
import numpy as np
import matplotlib.pyplot as plt


img_lena = np.double(imread('../../data/lena.tif'))
img_small = np.double(imread('../../data/lena_small.tif'))

print(img_lena.shape, img_small.shape)
bits = 3  # 3-bit 量化
epsilon = 1e-5  # Lloyd-Max

# 1.uniform
qImage_uni_lena = uniquant(img_lena, bits)
qImage_uni_small = uniquant(img_small, bits)

recImage_uni_lena = inv_uniquant(qImage_uni_lena, bits)
recImage_uni_small = inv_uniquant(qImage_uni_small, bits)

# 2. Lloyd-Max
qImage_lm_lena, clusters_lena = lloyd_max(img_lena, bits, epsilon)
qImage_lm_small, clusters_small = lloyd_max(img_small, bits, epsilon)

recImage_lm_lena = inv_lloyd_max(qImage_lm_lena, clusters_lena)
recImage_lm_small = inv_lloyd_max(qImage_lm_small, clusters_small)

# 3.  PSNR
# unifrom PSNR
PSNR_uni_small = calc_psnr(img_small, recImage_uni_small)
PSNR_uni_lena = calc_psnr(img_lena, recImage_uni_lena)

# Lloyd-Max  PSNR
PSNR_lm_small = calc_psnr(img_small, recImage_lm_small)
PSNR_lm_lena = calc_psnr(img_lena, recImage_lm_lena)


print("\nPSNR Comparison for 3-bit quantization:")
print("Small Image:")
print(f"Uniform Quantization PSNR: {PSNR_uni_small:.2f} dB")
print(f"Lloyd-Max Quantization PSNR: {PSNR_lm_small:.2f} dB")
print("\nLarge Image:")
print(f"Uniform Quantization PSNR: {PSNR_uni_lena:.2f} dB")
print(f"Lloyd-Max Quantization PSNR: {PSNR_lm_lena:.2f} dB")


plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(img_small.astype(int), cmap='gray')


plt.subplot(1, 3, 2)
plt.title("Uniform Quantization")
plt.imshow(recImage_uni_small.astype(int), cmap='gray')

plt.subplot(1, 3, 3)
plt.title("Lloyd-Max Quantization")
plt.imshow(recImage_lm_small.astype(int), cmap='gray')

plt.show()