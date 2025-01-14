import numpy as np
from ivclab.image import IntraCodec
from ivclab.utils import imread, calc_psnr
import matplotlib.pyplot as plt

# Implement the IntraCodec and all the necessary modules
# For each given quantization scale in the handout:
# - Initialize a new IntraCodec
# - Use lena_small to train Huffman coder of IntraCodec.
# - Compress and decompress 'lena.tif'
# - Measure bitrate and PSNR on lena
# Plot all the measurements in a Rate Distortion plot

# YOUR CODE STARTS HERE

# YOUR CODE ENDS HERE

import numpy as np
from ivclab.image import IntraCodec
from ivclab.utils import imread, calc_psnr
import matplotlib.pyplot as plt

# 1. 读取图像
lena_small = np.double(imread('../../data/lena_small.tif'))
lena = np.double(imread('../../data/lena.tif'))

# 2. 定义测试参数
q_scales = [0.15, 0.3, 0.7, 1.0, 1.5, 3, 5, 7, 10]  # 指定的量化尺度
bounds = (-1000, 4000)  # 指定的范围
psnr_values = []
bitrate_values = []

# 3. 对每个量化尺度进行测试
for scale in q_scales:
    # 创建编码器实例，使用指定的参数
    codec = IntraCodec(
        quantization_scale=scale,
        bounds=bounds,
        end_of_block=4000,
        block_shape=(8, 8)
    )

    # 训练Huffman编码器
    codec.train_huffman_from_image(lena_small)

    # 压缩和解压缩
    bitstream = codec.intra_encode(lena)
    reconstructed_img = codec.intra_decode(bitstream, lena.shape)

    # 计算PSNR和比特率
    psnr = calc_psnr(lena, reconstructed_img)
    total_bits = len(bitstream) * 8 * 4
    total_pixels = lena.shape[0] * lena.shape[1]
    bitrate = total_bits / total_pixels

    # 存储结果
    psnr_values.append(psnr)
    bitrate_values.append(bitrate)

    print(f"Scale {scale}: PSNR = {psnr:.2f} dB, Bitrate = {bitrate:.2f} bits/pixel")

# 4. 绘制Rate-Distortion曲线
plt.figure(figsize=(10, 6))
plt.plot(bitrate_values, psnr_values, 'bo-', linewidth=2, markersize=8)

# 在每个点上标注量化尺度
for i, scale in enumerate(q_scales):
    plt.annotate(f'q={scale}',
                 (bitrate_values[i], psnr_values[i]),
                 xytext=(5, 5),
                 textcoords='offset points')

# 5. 美化图表
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Bitrate (bits/pixel)')
plt.ylabel('PSNR (dB)')
plt.title('Rate-Distortion Curve')
plt.grid(True)

plt.show()