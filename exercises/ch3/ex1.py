from ivclab.utils import imread, calc_psnr
from ivclab.quantization.quantizers import uniquant, inv_uniquant
import numpy as np
import matplotlib.pyplot as plt

# read images
img_lena = np.double(imread('data/lena.tif'))
img_small = np.double(imread('data/lena_small.tif'))


# Initialize lists to store quantized images
qImage_lena = []
qImage_small = []

reconstructed_lena = []
reconstructed_small = []

# Bit depths to iterate over
bits = np.arange(1, 8)

# Perform quantization
for bit in bits:
    qImage_lena.append(uniquant(img_lena, bit))
    qImage_small.append(uniquant(img_small, bit))

for bit in bits:
    reconstructed_lena.append(inv_uniquant(qImage_lena[bit-1], bit))
    reconstructed_small.append(inv_uniquant(qImage_small[bit-1], bit))


# Plot results
    
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img_lena.astype(int), cmap='gray')
#plt.subplot(1, 3, 2)
#plt.title("Quantized Image")  # if not work, remove
#plt.matshow(qImage_lena[2].astype(int))
plt.subplot(1, 2, 2)
plt.title("Reconstructed Image")
plt.imshow(reconstructed_lena[2].astype(int), cmap='gray')
plt.show()



# Quantization bits for lena_small
bits_small = [1, 2, 3, 4, 5, 6, 7]
PSNR_small = []

# Quantize and compute PSNR for lena_small
for bits in bits_small:
    qImageLena_small = uniquant(img_small, bits)
    recImage = inv_uniquant(qImageLena_small, bits)
    PSNR_small.append(calc_psnr(img_small, recImage))


# Quantization bits for lena
bits = [1, 2, 3, 4, 5, 6, 7]
PSNR = []

# Quantize and compute PSNR for lena
for b in bits:
    qImageLena = uniquant(img_lena, b)
    recImage = inv_uniquant(qImageLena, b)
    PSNR.append(calc_psnr(img_lena, recImage))

# Print results
print("PSNR for lena_small:", PSNR_small)
print("PSNR for lena:", PSNR)