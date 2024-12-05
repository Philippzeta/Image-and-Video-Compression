from ivclab import HuffmanCoder
from ivclab.utils import imread, calc_psnr
from ivclab.signal import rgb2ycbcr, ycbcr2rgb
from ivclab.entropy import stats_marg
from ivclab.image import yuv420_three_pixel_residual, reconstruct_ycbcr_from_residuals
import numpy as np
import matplotlib.pyplot as plt


# 读取并压缩原始图像
image = imread('../../data/lena.tif')
image = image.astype(np.float32)
ycbcr_image = rgb2ycbcr(image)

# 获取预测残差
residual_image_Y, residual_image_CbCr = yuv420_three_pixel_residual(ycbcr_image)

# 打印形状信息以便调试
print("Original image shape:", image.shape)
print("Y residual shape:", residual_image_Y.shape)
print("CbCr residual shape:", residual_image_CbCr.shape)

# 合并残差数据用于Huffman编码
merged_residuals = np.concatenate([residual_image_Y.ravel(), residual_image_CbCr.ravel()])

# 将残差转换为整数类型
merged_residuals = np.round(merged_residuals).astype(np.int32)

# Huffman编码过程
pmf = stats_marg(merged_residuals, np.arange(-255, 256))
huffman_coder = HuffmanCoder(lower_bound=-255)
huffman_coder.train(pmf)
compressed, bitrate = huffman_coder.encode(merged_residuals)

# Huffman解码
decoded_residuals = huffman_coder.decode(compressed, message_length=len(merged_residuals))

# Reconstruction
H, W, _ = ycbcr_image.shape
Y_size = residual_image_Y.size  # 使用实际的Y残差大小
reconstructed_ycbcr = reconstruct_ycbcr_from_residuals(decoded_residuals, ycbcr_image.shape)

# Convert YCbCr back to RGB
reconstructed_rgb = ycbcr2rgb(reconstructed_ycbcr)

# 评估和展示结果
plt.figure(figsize=(12, 5))

# 显示原始图像
plt.subplot(121)
plt.imshow(image.astype(np.uint8))
plt.title('Original Image')
plt.axis('off')

# 显示重构图像
PSNR = calc_psnr(image, reconstructed_rgb)
plt.subplot(122)
plt.imshow(reconstructed_rgb.astype(np.uint8))
plt.title(f'Reconstructed Image\nPSNR = {PSNR:.2f} dB')
plt.axis('off')

plt.tight_layout()

# 计算和打印评估指标
BPP = len(compressed) * 8 / (image.size / 3)
CompressionRatio = 24 / BPP

print(f'Bit Rate         = {BPP:.2f} bit/pixel')
print(f'CompressionRatio = {CompressionRatio:.2f}')
print(f'PSNR             = {PSNR:.2f} dB')

plt.show()