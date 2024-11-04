import numpy as np
import matplotlib.pyplot as plt
from ivclab.utils import imread, calc_psnr

orig_img = imread('../../data/smandril.tif')
recon_img = imread('../../data/smandril_rec.tif')

bitrate_original = 24
bitrate_compressed = 8

psnr = calc_psnr(orig_img, recon_img)

bitrates = [bitrate_original, bitrate_compressed]
psnrs = [np.nan, psnr]

plt.figure(figsize=(10, 6))
plt.plot(bitrates, psnrs, 'bo-', label='Rate-Distortion Curve')
plt.xlabel('Bit Rate (bits per pixel)')
plt.ylabel('PSNR (dB)')
plt.title('Rate vs Distortion Curve (Bit Rate vs PSNR)')
plt.grid(True)
plt.legend()
plt.show()
