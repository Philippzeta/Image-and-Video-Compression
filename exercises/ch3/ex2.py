from ivclab.utils import imread, calc_psnr
from ivclab.quantization.quantizers import lloyd_max, inv_lloyd_max
import numpy as np
import matplotlib.pyplot as plt

# Example input image
img_lena = np.double(imread('data/lena.tif'))
img_small = np.double(imread('data/lena_small.tif'))
print(img_lena.shape, img_small.shape)
bits = 3  # Example bit depth
epsilon = 1e-5  # Convergence threshold

# Perform Lloyd-Max quantization
qImage_lena, clusters_lena = lloyd_max(img_lena, bits, epsilon)
qImage_small, clusters_small = lloyd_max(img_small, bits, epsilon)

# Display results
print("Clusters Lena (quantization levels):", clusters_lena)

recImage_lena = inv_lloyd_max(qImage_lena, clusters_lena)
recImage_small = inv_lloyd_max(qImage_small, clusters_small)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img_small.astype(int), cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Reconstructed Image")
plt.imshow(recImage_small.astype(int), cmap='gray')
plt.show()



PSNR_small = calc_psnr(img_small, recImage_small)
PSNR = calc_psnr(img_lena, recImage_lena)

# Print Results
print(f"PSNR for lena_small: {PSNR_small:.2f} dB")
print(f"PSNR for lena: {PSNR:.2f} dB")