from ivclab.utils import imread, imshow
from ivclab.signal import rgb2gray
import matplotlib.pyplot as plt

# read images
img_lena = imread('../../data/lena.tif')
img_lena_gray = rgb2gray(img_lena)
img_smandril = imread('../../data/smandril.tif')
img_smandril_gray = rgb2gray(img_smandril)

# E1-1 b Calculate compression ratio
#compression_ratio_lena = img_lena.shape[-1] / img_lena_gray.shape[-1]
#compression_ratio_smandril = img_smandril.shape[-1] / img_smandril_gray.shape[-1]

# Print compression ratios
#print(f'Compression ratio for Lena image: {compression_ratio_lena}')
#print(f'Compression ratio for Smandril image: {compression_ratio_smandril}')

# plot images
fig, axs = plt.subplots(2,2)
imshow(axs[0][0], img_lena, title='Original Lena Image')
imshow(axs[0][1], img_lena_gray, title='Compressed Lena Image')
imshow(axs[1][0], img_smandril, title='Original Smandril Image')
imshow(axs[1][1], img_smandril_gray, title='Compressed Smandril Image')

plt.show()


