import numpy as np
from ivclab.image import IntraCodec
from ivclab.utils import imread, calc_psnr
# Implement the IntraCodec and all the necessary modules
# Use lena_small to train Huffman coder of IntraCodec.
# Use it to compress and decompress 'lena.tif'
# Measure bitrate and PSNR on lena

# YOUR CODE STARTS HERE
# 1. 读取训练和测试图像
lena_small = imread(f'../../data/lena_small.tif')
lena = imread(f'../../data/lena.tif')

# 2. 创建IntraCodec实例
codec = IntraCodec(
    quantization_scale=8.0,
    bounds=(-20, 4000),
    end_of_block=4000,
    block_shape=(8, 8)
)

# 3. 使用lena_small训练Huffman编码器
train_symbols = codec.image2symbols(lena_small)
print(f"Training symbols range: [{min(train_symbols)}, {max(train_symbols)}]")
print(f"Number of unique symbols in training: {len(set(train_symbols))}")

codec.train_huffman_from_image(lena_small)

# 4. 压缩lena图像得到比特流
test_symbols = codec.image2symbols(lena)
print(f"Test symbols range: [{min(test_symbols)}, {max(test_symbols)}]")
print(f"Number of unique symbols in test: {len(set(test_symbols))}")

bitstream = codec.intra_encode(lena)

# 5. 从比特流解码重建图像
reconstructed_img = codec.intra_decode(bitstream, lena.shape)

# 6. 计算PSNR
psnr = calc_psnr(lena, reconstructed_img)

# 7. 计算比特率
total_bits = len(bitstream)
total_pixels = lena.shape[0] * lena.shape[1] * lena.shape[2]
bitrate = total_bits / total_pixels

print(f"PSNR: {psnr:.2f} dB")
print(f"Bitrate: {bitrate:.2f} bits/pixel")
# YOUR CODE ENDS HERE
