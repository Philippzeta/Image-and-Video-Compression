from ivclab import HuffmanCoder
from ivclab.utils import imread,calc_psnr
from ivclab.entropy import stats_marg
from ivclab.image import three_pixels_predictor,yuv420compression,inverse_three_pixels_predictor
import numpy as np


# 读取并压缩原始图像
image = imread('../../data/lena.tif')
recon_image = yuv420compression(image)

# 获取预测残差
residual_image_Y, residual_image_CbCr = three_pixels_predictor(recon_image, subsample_color_channels=False)

# 打印原始残差形状
print("Original Y shape:", residual_image_Y.shape)
print("Original CbCr shape:", residual_image_CbCr.shape)

# 合并残差数据用于Huffman编码
merged_residuals = np.concatenate([residual_image_Y.ravel(), residual_image_CbCr.ravel()])

# Huffman编码过程
pmf = stats_marg(merged_residuals, np.arange(-255,256))
huffman_coder = HuffmanCoder(lower_bound=-255)
huffman_coder.train(pmf)
compressed, bitrate = huffman_coder.encode(merged_residuals)

# Huffman解码
decoded_residuals = huffman_coder.decode(compressed, message_length=len(merged_residuals))

# 将解码后的一维数组重新分割为Y和CbCr残差
y_size = residual_image_Y.size
height, width = residual_image_Y.shape[:2]  # 获取实际的图像尺寸
decoded_residual_Y = decoded_residuals[:y_size].reshape(height, width, 1)  # reshape为[H, W, 1]
decoded_residual_CbCr = decoded_residuals[y_size:].reshape(height, width, 2)  # reshape为[H, W, 2]

# 打印解码后的形状
print("Decoded Y shape:", decoded_residual_Y.shape)
print("Decoded CbCr shape:", decoded_residual_CbCr.shape)
print("Original image shape:", recon_image.shape)

# 在重建图像之前添加这些调试信息
print("\nDebug information:")
print(f"decoded_residual_Y min: {decoded_residual_Y.min()}, max: {decoded_residual_Y.max()}")
print(f"decoded_residual_CbCr min: {decoded_residual_CbCr.min()}, max: {decoded_residual_CbCr.max()}")
print(f"decoded_residual_Y dtype: {decoded_residual_Y.dtype}")
print(f"decoded_residual_CbCr dtype: {decoded_residual_CbCr.dtype}")
print(f"decoded_residual_Y shape: {decoded_residual_Y.shape}")
print(f"decoded_residual_CbCr shape: {decoded_residual_CbCr.shape}")
print(f"recon_image min: {recon_image.min()}, max: {recon_image.max()}")

# 确保数据类型正确
decoded_residual_Y = decoded_residual_Y.astype(np.float32)
decoded_residual_CbCr = decoded_residual_CbCr.astype(np.float32)

reconstructed_image = inverse_three_pixels_predictor(
    decoded_residual_Y,
    decoded_residual_CbCr,
    original_shape=recon_image.shape,
    subsample_color_channels=False
)
psnr = calc_psnr(recon_image, reconstructed_image)
print(f"PSNR: {psnr:.2f} dB")
print(f"Compression bitrate: {bitrate:.2f} bits")

# 在获取预测残差后添加调试信息
print("\nOriginal residuals information:")
print(f"residual_image_Y min: {residual_image_Y.min()}, max: {residual_image_Y.max()}")
print(f"residual_image_CbCr min: {residual_image_CbCr.min()}, max: {residual_image_CbCr.max()}")

# 在重建后添加调试信息
print("\nReconstructed information:")
print(f"reconstructed_image min: {reconstructed_image.min()}, max: {reconstructed_image.max()}")
print(f"recon_image - reconstructed_image mean: {np.mean(np.abs(recon_image - reconstructed_image))}")


# 计算压缩率
original_size = image.size * 8  # 原始图像大小（位）
compressed_size = len(compressed) * 8  # 压缩后大小（位）
compression_ratio = original_size / compressed_size

print("\nCompression Statistics:")
print(f"Original size: {original_size} bits")
print(f"Compressed size: {compressed_size} bits")
print(f"Compression ratio: {compression_ratio:.2f}:1")
print(f"Bits per pixel: {compressed_size / (height * width * 3):.2f}")

