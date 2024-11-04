from ivclab.utils import imread, calc_mse, calc_psnr

# Read the original and reconstructed images
orig_img = imread('../../data/smandril.tif')  # org
recon_img = imread('../../data/smandril_rec.tif')  # recon

# Calculate MSE and PSNR
mse = calc_mse(orig_img, recon_img)
psnr = calc_psnr(orig_img, recon_img)

# Print the results
print(f'PSNR: {psnr:.4f}')
print(f'MSE: {mse:.4f}')
