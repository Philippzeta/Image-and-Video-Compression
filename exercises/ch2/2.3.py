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