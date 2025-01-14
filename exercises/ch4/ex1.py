import numpy as np
import matplotlib.pyplot as plt
from ivclab.image import IntraCodec
from ivclab.utils import imread, calc_psnr,imshow
# Implement the IntraCodec and all the necessary modules
# Use lena_small to train Huffman coder of IntraCodec.
# Use it to compress and decompress 'lena.tif'
# Measure bitrate and PSNR on lena

# YOUR CODE STARTS HERE
# 1. 读取训练和测试图像
lena_small = np.double(imread(f'../../data/lena_small.tif'))
lena = np.double(imread(f'../../data/lena.tif'))
# 2
codec = IntraCodec()
# 3. 使用lena_small训练Huffman编码器
train_symbols = codec.image2symbols(lena_small)
codec.train_huffman_from_image(lena_small)

# 4. 压缩lena图像得到比特流
test_symbols = codec.image2symbols(lena)
bitstream = codec.intra_encode(lena)

# 5. 从比特流解码重建图像
reconstructed_img = codec.intra_decode(bitstream, lena.shape)

# 6. 计算PSNR
psnr = calc_psnr(lena, reconstructed_img)

# 7. 计算比特率
total_bits = len(bitstream)*8*4
total_pixels = lena.shape[0] * lena.shape[1]
bitrate = total_bits / total_pixels

print(f"PSNR: {psnr:.2f} dB")
print(f"Bitrate: {bitrate:.2f} bits/pixel")
# YOUR CODE ENDS HERE

plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.imshow(lena/255.0)  # 归一化
plt.title('Original Image')
plt.axis('off')

plt.subplot(122)
plt.imshow(reconstructed_img/255.0)  # 归一化
plt.title(f'Reconstructed (PSNR: {psnr:.2f}dB)')
plt.axis('off')

plt.tight_layout()
plt.show()