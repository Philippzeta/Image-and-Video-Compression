# Read fore0020.bmp and compress it with the intra codec trained on lena_small.
# Report the PSNR and bpp of the compressed image.
# Use q_scale of 0.15 and bounds of (-1000, 4000).
import numpy as np
from ivclab.image import IntraCodec
from ivclab.utils import imread, calc_psnr

# 1. 读取训练和测试图像
lena_small = np.double(imread(f'../../data/lena_small.tif'))
foreman0022 = np.double(imread(f'../../videodata/foreman0022.bmp'))
# 2
codec = IntraCodec()
# 3. 使用lena_small训练Huffman编码器
train_symbols = codec.image2symbols(lena_small)
codec.train_huffman_from_image(lena_small)

# 4. 压缩lena图像得到比特流
test_symbols = codec.image2symbols(foreman0022)
bitstream = codec.intra_encode(foreman0022)

# 5. 从比特流解码重建图像
reconstructed_img = codec.intra_decode(bitstream, foreman0022.shape)

# 6. 计算PSNR
psnr = calc_psnr(foreman0022, reconstructed_img)

# 7. 计算比特率
total_bits = len(bitstream)*8*4
total_pixels = foreman0022.shape[0] * foreman0022.shape[1]
bitrate = total_bits / total_pixels

print(f"PSNR: {psnr:.2f} dB")
print(f"Bitrate: {bitrate:.2f} bits/pixel")