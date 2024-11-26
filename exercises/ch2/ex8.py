from ivclab import HuffmanCoder
from ivclab.utils import imread,calc_psnr
from ivclab.entropy import stats_marg
from ivclab.image import three_pixels_predictor
import numpy as np
import matplotlib.pyplot as plt
from ivclab.image import yuv420compression

image = imread('../../data/lena_small.tif')
recon_image = yuv420compression(image)
psnr_recon = calc_psnr(image, recon_image)
print(f"Reconstructed image, not prefiltered, PSNR = {psnr_recon:.2f} dB")

residual_image_Y, residual_image_CbCr = three_pixels_predictor(recon_image, subsample_color_channels=False)
merged_residuals = np.concatenate([residual_image_Y.ravel(), residual_image_CbCr.ravel()])
pmf = stats_marg(merged_residuals, np.arange(-255,256))
huffman_coder = HuffmanCoder(lower_bound=-255)
huffman_coder.train(pmf)
compressed, bitrate = huffman_coder.encode(merged_residuals)
print(f"Compressed: {compressed}")
print(f"Bitrate: {bitrate}")
decoded_message = huffman_coder.decode(compressed, message_length=len(merged_residuals))
#print(np.array_equal(decoded_message, merged_residuals))

