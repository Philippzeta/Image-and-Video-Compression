import numpy as np
def quantize_dct_coefficients(dct_blocks_y, dct_blocks_cb, dct_blocks_cr):

    L = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 55, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    C = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 13, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ])
    def quantize_block(block, qtable):
        return np.round(block / qtable)

    quant_y = np.array([quantize_block(block, L) for block in dct_blocks_y])
    quant_cb = np.array([quantize_block(block, C) for block in dct_blocks_cb])
    quant_cr = np.array([quantize_block(block, C) for block in dct_blocks_cr])

    return quant_y, quant_cb, quant_cr


def dequantize_coefficients(quant_y, quant_cb, quant_cr):

    L = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 55, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    C = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 13, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ])

    def dequantize_block(block, qtable):
        return block * qtable
    dequant_y = np.array([dequantize_block(block, L) for block in quant_y])
    dequant_cb = np.array([dequantize_block(block, C) for block in quant_cb])
    dequant_cr = np.array([dequantize_block(block, C) for block in quant_cr])

    return dequant_y, dequant_cb, dequant_cr


# 测试代码
def test_quantization():
    # 创建示例DCT系数
    test_block_y = np.random.randn(1, 8, 8) * 100
    test_block_cb = np.random.randn(1, 8, 8) * 100
    test_block_cr = np.random.randn(1, 8, 8) * 100

    # 量化
    quant_y, quant_cb, quant_cr = quantize_dct_coefficients(
        test_block_y, test_block_cb, test_block_cr)

    print("原始Y分量DCT系数:")
    print(test_block_y[0])
    print("\n量化后的Y分量系数:")
    print(quant_y[0])


if __name__ == "__main__":
    test_quantization()