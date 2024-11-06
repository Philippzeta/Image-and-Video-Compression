from ivclab.utils import imread, calc_psnr
from ivclab.image import yuv420compression
import numpy as np
import matplotlib.pyplot as plt

image = imread('../../data/sail.tif')

recon_image = yuv420compression(image)

psnr_recon = calc_psnr(image, recon_image)


# Calculate bitrate based on the compressed image
compressed_size = recon_image.size * 8  # Reconstructed image size in bits (assuming 8 bits per channel)

# Bitrate calculation (bits per pixel)
bitrate = compressed_size / (image.shape[0] * image.shape[1])

print(f"Reconstructed image, not prefiltered, PSNR = {psnr_recon:.2f} dB")
print(f"Bitrate of the compressed image = {bitrate:.2f} bits per pixel")

# Plot original and reconstructed images side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Original image
axes[0].imshow(image)
axes[0].set_title('Original Image')
axes[0].axis('off')

# Reconstructed image
axes[1].imshow(recon_image)
axes[1].set_title('Reconstructed Image')
axes[1].axis('off')

plt.tight_layout()
plt.show()
