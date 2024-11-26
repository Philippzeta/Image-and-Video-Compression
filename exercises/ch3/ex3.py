import numpy as np
from ivclab.quantization.quantizers import vector_quantizer, apply_vector_quantizer, inv_vector_quantizer
from ivclab.utils import calc_psnr, imread
from ivclab.entropy.huffman import HuffmanCoder
import matplotlib.pyplot as plt

# Image Loading (in Python)
img_small = np.double(imread('data/lena_small.tif'))
bits = 8
epsilon = 0.1
block_size = 2


# Process small image (lena_small.tif)
clusters, _ = vector_quantizer(img_small, bits, epsilon, block_size)
qImage_small = apply_vector_quantizer(img_small, clusters, block_size)

# Huffman encoding and quantization
lower_bound = 0
upper_bound = 2**bits
val = np.arange(lower_bound, upper_bound + 1)
pmf = np.histogram(qImage_small.flatten(), bins=val, density=True)[0]

# Build Huffman codes
huffman_coder = HuffmanCoder(lower_bound=lower_bound)
huffman_coder.train(probs=pmf)

# Process large image (lena.tif)
img_lena = np.double(imread('data/lena.tif'))
qImage = apply_vector_quantizer(img_lena, clusters, block_size)

# Huffman Encoding and Decoding
bytestream, bitrate = huffman_coder.encode(qImage.flatten())
k_rec = huffman_coder.decode(bytestream, qImage.size) 

# Reconstruct the image
qReconst_image = k_rec.reshape(qImage.shape)
reconst_image = inv_vector_quantizer(qReconst_image, clusters, block_size)

# Calculate PSNR
PSNR_value = calc_psnr(img_lena, reconst_image)
print("PSNR:", PSNR_value)

# Bit per pixel calculation (bpp)
bpp = (len(bytestream) * 8) / (img_lena.size / 3)
print(img_lena.size)
print("Bits per pixel:", bpp)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img_lena.astype(int))
plt.subplot(1, 2, 2)
plt.title("Reconstructed Image")
plt.imshow(reconst_image.astype(int))
plt.show()