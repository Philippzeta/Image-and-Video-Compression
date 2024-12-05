import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
from ivclab import HuffmanCoder
from ivclab.utils import imread, calc_psnr
from ivclab.signal import rgb2ycbcr, ycbcr2rgb
from ivclab.entropy import stats_marg

# 读取图像
image = imread('../../data/lena.tif').astype(np.float32)
ycbcr_image = rgb2ycbcr(image)

# Wrap and Downsample CbCr
pad_pixel = 4
H, W, _ = ycbcr_image.shape
I_CbCr_wrapped = np.zeros((H + 2 * pad_pixel, W + 2 * pad_pixel, 2))
I_CbCr_down = np.zeros((H // 2, W // 2, 2))  # 修正: 确保大小是原始图像的一半

for i in range(2):
    # 填充
    I_CbCr_wrapped[:, :, i] = np.pad(ycbcr_image[:, :, i + 1],
                                     ((pad_pixel, pad_pixel), (pad_pixel, pad_pixel)),
                                     mode='symmetric')
    # 降采样
    wrapped = resample_poly(I_CbCr_wrapped[:, :, i], 1, 2, axis=0)
    wrapped = resample_poly(wrapped, 1, 2, axis=1)
    # 裁剪到正确的大小
    I_CbCr_down[:, :, i] = wrapped[pad_pixel // 2:pad_pixel // 2 + H // 2,
                           pad_pixel // 2:pad_pixel // 2 + W // 2]

# 初始化残差和重构数组
sz_CbCr = I_CbCr_down.shape
resImage_Y = np.zeros((H, W), dtype=np.int32)  # 修正: 使用整数类型
recImage_Y = np.zeros((H, W))
resImage_Y[:, 0] = ycbcr_image[:, 0, 0]  # copy first column
resImage_Y[0, :] = ycbcr_image[0, :, 0]  # copy first row
recImage_Y[:, 0] = ycbcr_image[:, 0, 0]
recImage_Y[0, :] = ycbcr_image[0, :, 0]

# 预测系数（与MATLAB相同）
coeff_luminance = np.array([[-1 / 2, 5 / 8], [7 / 8, 0]])
coeff_chrominance = np.array([[-1 / 2, 7 / 8], [5 / 8, 0]])

# Y通道预测
for row in range(1, H):
    for col in range(1, W):
        recPatch = recImage_Y[row - 1:row + 1, col - 1:col + 1]
        prediction = np.sum(coeff_luminance * recPatch)
        resImage_Y[row, col] = round(ycbcr_image[row, col, 0] - prediction)
        recImage_Y[row, col] = prediction + resImage_Y[row, col]

# CbCr通道预测
resImage_CbCr = np.zeros((sz_CbCr[0], sz_CbCr[1], 2), dtype=np.int32)  # 修正: 使用整数类型
recImage_CbCr = np.zeros((sz_CbCr[0], sz_CbCr[1], 2))
for ch in range(2):
    resImage_CbCr[:, 0, ch] = I_CbCr_down[:, 0, ch]
    resImage_CbCr[0, :, ch] = I_CbCr_down[0, :, ch]
    recImage_CbCr[:, 0, ch] = I_CbCr_down[:, 0, ch]
    recImage_CbCr[0, :, ch] = I_CbCr_down[0, :, ch]

    for row in range(1, sz_CbCr[0]):
        for col in range(1, sz_CbCr[1]):
            recPatch = recImage_CbCr[row - 1:row + 1, col - 1:col + 1, ch]
            prediction = np.sum(coeff_chrominance * recPatch)
            resImage_CbCr[row, col, ch] = round(I_CbCr_down[row, col, ch] - prediction)
            recImage_CbCr[row, col, ch] = prediction + resImage_CbCr[row, col, ch]

# 合并所有残差为一个数组
combined_residuals = np.concatenate([
    resImage_Y.ravel(),
    resImage_CbCr[:, :, 0].ravel(),
    resImage_CbCr[:, :, 1].ravel()
]).astype(np.int32)

# Huffman编码
range_vals = np.arange(-128, 256)
pmf = stats_marg(combined_residuals, range_vals)
huffman_coder = HuffmanCoder(lower_bound=-128)
huffman_coder.train(pmf)

# 编码整个combined_residuals
encoded_data = huffman_coder.encode(combined_residuals)
bytestream = encoded_data[0]  # 使用第一个元素作为bytestream

# 解码
decoded_residuals = np.array(huffman_coder.decode(bytestream, message_length=combined_residuals.size))

# 确保解码后的大小正确
assert decoded_residuals.size == combined_residuals.size, "Decoded size mismatch"

# 分离解码后的残差
Y_size = resImage_Y.size
CbCr_size = resImage_CbCr[:, :, 0].size

decoded_residuals_Y = decoded_residuals[:Y_size].reshape(H, W)
decoded_residuals_Cb = decoded_residuals[Y_size:Y_size + CbCr_size].reshape(sz_CbCr[0], sz_CbCr[1])
decoded_residuals_Cr = decoded_residuals[Y_size + CbCr_size:].reshape(sz_CbCr[0], sz_CbCr[1])

# 重构Y通道
decRecImage_Y = np.zeros((H, W))
decRecImage_Y[:, 0] = ycbcr_image[:, 0, 0]
decRecImage_Y[0, :] = ycbcr_image[0, :, 0]

for row in range(1, H):
    for col in range(1, W):
        recPatch = decRecImage_Y[row - 1:row + 1, col - 1:col + 1]
        prediction = np.sum(coeff_luminance * recPatch)
        decRecImage_Y[row, col] = prediction + decoded_residuals_Y[row, col]

# 重构CbCr通道
decRecImage_CbCr = np.zeros_like(I_CbCr_down)
for ch in range(2):
    decRecImage_CbCr[:, 0, ch] = I_CbCr_down[:, 0, ch]
    decRecImage_CbCr[0, :, ch] = I_CbCr_down[0, :, ch]

    for row in range(1, sz_CbCr[0]):
        for col in range(1, sz_CbCr[1]):
            recPatch = decRecImage_CbCr[row - 1:row + 1, col - 1:col + 1, ch]
            prediction = np.sum(coeff_chrominance * recPatch)
            if ch == 0:
                decRecImage_CbCr[row, col, ch] = prediction + decoded_residuals_Cb[row, col]
            else:
                decRecImage_CbCr[row, col, ch] = prediction + decoded_residuals_Cr[row, col]

# Upsample
I_CbCr_wrapped_2 = np.pad(decRecImage_CbCr,
                          ((pad_pixel // 2, pad_pixel // 2), (pad_pixel // 2, pad_pixel // 2), (0, 0)),
                          mode='symmetric')
I_CbCr_up = np.zeros((H, W, 2))
for i in range(2):
    wrapped = resample_poly(I_CbCr_wrapped_2[:, :, i], 2, 1, axis=0)
    wrapped = resample_poly(wrapped, 2, 1, axis=1)
    I_CbCr_up[:, :, i] = wrapped[pad_pixel:-pad_pixel, pad_pixel:-pad_pixel]

# 合并通道
reconstructed_ycbcr = np.dstack((decRecImage_Y, I_CbCr_up))
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
BPP = len(bytestream) * 8 / (image.size / 3)
CompressionRatio = 24 / BPP

print(f'Bit Rate         = {BPP:.2f} bit/pixel')
print(f'CompressionRatio = {CompressionRatio:.2f}')
print(f'PSNR             = {PSNR:.2f} dB')

plt.show()