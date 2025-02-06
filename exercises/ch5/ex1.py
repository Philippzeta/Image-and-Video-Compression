# Read fore0020.bmp and compress it with the intra codec trained on lena_small.
# Report the PSNR and bpp of the compressed image.
# Use q_scale of 0.15 and bounds of (-1000, 4000).
import numpy as np
import matplotlib.pyplot as plt
from ivclab.image import IntraCodec
from ivclab.utils import imread, calc_psnr

# 1. read and train
lena_small = np.double(imread(f'../../data/lena_small.tif'))
foreman0020 = np.double(imread(f'../../videodata/foreman0020.bmp.'))
codec = IntraCodec()
# 3. use lena_small train
train_symbols = codec.image2symbols(lena_small)
codec.train_huffman_from_image(lena_small)

# 4. get bitstream
test_symbols = codec.image2symbols(foreman0020)
bitstream = codec.intra_encode(foreman0020)

# 5. recon
reconstructed_img = codec.intra_decode(bitstream, foreman0020.shape)

# 6. PSNR
psnr = calc_psnr(foreman0020, reconstructed_img)

# 7. bbp
total_bits = len(bitstream)*8
total_pixels = foreman0020.shape[0] * foreman0020.shape[1]
pixel_number = total_pixels/4
bitrate = total_bits /pixel_number

print(f"PSNR: {psnr:.2f} dB")
print(f"Bitrate: {bitrate:.2f} bits/pixel")

plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.imshow(foreman0020/255.0)  # NOR
plt.title('Original Image')
plt.axis('off')

plt.subplot(122)
plt.imshow(reconstructed_img/255.0)  # NOR
plt.title(f'Reconstructed (PSNR: {psnr:.2f}dB)')
plt.axis('off')

plt.tight_layout()
plt.show()