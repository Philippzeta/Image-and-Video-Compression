from ivclab.utils import imread, calc_psnr
import numpy as np
from scipy.signal import resample

# rgb function
def rgb_compression(image: np.ndarray):

    # Step 1: Compress the image
    # A. Pad the image with 4 pixels symmetric on each side
    padded_image = np.pad(image, ((4, 4), (4, 4), (0, 0)), mode='symmetric')

    # B. Downsample RGB channels with prefiltering
    R_downsampled = resample(padded_image[:, :, 0], padded_image.shape[0] // 2, axis=0)
    R_downsampled = resample(R_downsampled, padded_image.shape[1] // 2, axis=1)
    G_downsampled = resample(padded_image[:, :, 1], padded_image.shape[0] // 2, axis=0)
    G_downsampled = resample(G_downsampled, padded_image.shape[1] // 2, axis=1)
    B_downsampled = resample(padded_image[:, :, 2], padded_image.shape[0] // 2, axis=0)
    B_downsampled = resample(B_downsampled, padded_image.shape[1] // 2, axis=1)

    # C. Crop the image 2 pixels from each side to get rid of padding
    R_cropped = R_downsampled[2:-2, 2:-2]
    G_cropped = G_downsampled[2:-2, 2:-2]
    B_cropped = B_downsampled[2:-2, 2:-2]

    # Step 2: Apply rounding to RGB channels
    R_cropped = np.round(R_cropped)
    G_cropped = np.round(G_cropped)
    B_cropped = np.round(B_cropped)
    # Step 3: Decompress the image
    # A. Pad the image with 2 pixels symmetric on each side
    R_padded = np.pad(R_cropped, ((2, 2), (2, 2)), mode='symmetric')
    G_padded = np.pad(G_cropped, ((2, 2), (2, 2)), mode='symmetric')
    B_padded = np.pad(B_cropped, ((2, 2), (2, 2)), mode='symmetric')

    # B. Upsample RGB channels
    R_upsampled = resample(R_padded, padded_image.shape[0], axis=0)
    R_upsampled = resample(R_upsampled, padded_image.shape[1], axis=1)
    G_upsampled = resample(G_padded, padded_image.shape[0], axis=0)
    G_upsampled = resample(G_upsampled, padded_image.shape[1], axis=1)
    B_upsampled = resample(B_padded, padded_image.shape[0], axis=0)
    B_upsampled = resample(B_upsampled, padded_image.shape[1], axis=1)

    # C. Crop the image 4 pixels from each side to get rid of padding
    R_final = R_upsampled[4:-4, 4:-4]
    G_final = G_upsampled[4:-4, 4:-4]
    B_final = B_upsampled[4:-4, 4:-4]

    # Combine RGB channels to form the decompressed image
    decompressed_image = np.zeros_like(image)
    decompressed_image[:, :, 0] = R_final
    decompressed_image[:, :, 1] = G_final
    decompressed_image[:, :, 2] = B_final

    # Cast output to integer again
    decompressed_image = np.round(decompressed_image).astype(np.uint8)
    return decompressed_image



sail_image = imread('../../data/sail.tif')
sail_recon_image = rgb_compression(sail_image)
sail_psnr_recon = calc_psnr(sail_image, sail_recon_image)


lena_image = imread('../../data/lena.tif')
lena_recon_image = rgb_compression(lena_image)
lena_psnr_recon = calc_psnr(lena_image, lena_recon_image)

# Print the results
print("'sail.tif' reconstructed:  PSNR = {:.2f} dB".format(sail_psnr_recon))
print("'lena.tif' reconstructed:  PSNR = {:.2f} dB".format(lena_psnr_recon))
