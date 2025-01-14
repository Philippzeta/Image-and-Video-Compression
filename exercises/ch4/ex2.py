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
# 1. 读取图像
lena_small = imread(f'../../data/lena_small.tif')
lena = imread(f'../../data/lena.tif')

# 2. 定义不同的量化尺度
quantization_scales = [0.25, 0.5, 1.0, 2.0, 4.0]  # 从小到大的量化尺度
results = []

# 3. 对每个量化尺度进行测试
for scale in quantization_scales:
    # 创建新的编解码器
    codec = IntraCodec(
        quantization_scale=scale,
        bounds=(-1000, 4000),
        end_of_block=4000,
        block_shape=(8, 8)
    )

    # 训练Huffman编码器
    codec.train_huffman_from_image(lena_small)

    # 压缩和解压图像
    bitstream = codec.intra_encode(lena)
    reconstructed = codec.intra_decode(bitstream, lena.shape)

    # 计算PSNR和比特率
    psnr = calc_psnr(lena, reconstructed)
    total_bits = len(bitstream)
    total_pixels = lena.shape[0] * lena.shape[1] * lena.shape[2]
    bitrate = total_bits / total_pixels

    # 保存结果
    results.append({
        'scale': scale,
        'psnr': psnr,
        'bitrate': bitrate
    })

    print(f"Scale: {scale:.2f}")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"Bitrate: {bitrate:.2f} bits/pixel\n")

# 4. 绘制率失真曲线
plt.figure(figsize=(10, 6))
bitrates = [r['bitrate'] for r in results]
psnrs = [r['psnr'] for r in results]

plt.plot(bitrates, psnrs, 'bo-', linewidth=2, markersize=8)
plt.grid(True)
plt.xlabel('Bitrate (bits/pixel)')
plt.ylabel('PSNR (dB)')
plt.title('Rate-Distortion Curve for JPEG-like Compression')

# 在每个点上标注量化尺度
for r in results:
    plt.annotate(f'QS={r["scale"]:.2f}',
                 (r['bitrate'], r['psnr']),
                 xytext=(10, 10),
                 textcoords='offset points')

plt.show()
# YOUR CODE ENDS HERE
