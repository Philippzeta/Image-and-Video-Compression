from ivclab.utils import imread, calc_psnr
from ivclab.image import yuv420compression
import numpy as np
import matplotlib.pyplot as plt

image = imread('../../data/sail.tif')

recon_image = yuv420compression(image)

psnr_recon = calc_psnr(image, recon_image)


print(f"Reconstructed image, not prefiltered, PSNR = {psnr_recon:.2f} dB")


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
