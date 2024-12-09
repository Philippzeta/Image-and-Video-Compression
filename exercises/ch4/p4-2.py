import numpy as np

def zigzag_scan(matrix):
    if matrix.shape != (8, 8):
        raise ValueError("8*8!!")

    zigzag_coords = [
        (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
        (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
        (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
        (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
        (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
        (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
        (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
        (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)
    ]

    result = [matrix[i][j] for i, j in zigzag_coords]

    return np.array(result)


def inverse_zigzag_scan(zigzag):

    if len(zigzag) != 64:
        raise ValueError("length must be64")

    result = np.zeros((8, 8))
    zigzag_coords = [
        (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
        (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
        (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
        (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
        (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
        (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
        (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
        (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)
    ]

    for i, (row, col) in enumerate(zigzag_coords):
        result[row, col] = zigzag[i]

    return result


# 测试代码
def test_zigzag():
    # 创建测试矩阵
    test_matrix = np.arange(64).reshape(8, 8)
    print("原始矩阵:")
    print(test_matrix)

    # 进行之字形扫描
    zigzag = zigzag_scan(test_matrix)
    print("\n之字形扫描结果:")
    print(zigzag)

    # 重建矩阵
    reconstructed = inverse_zigzag_scan(zigzag)
    print("\n重建矩阵:")
    print(reconstructed)

    # 验证重建是否正确
    print("\n重建是否正确:", np.array_equal(test_matrix, reconstructed))


if __name__ == "__main__":
    test_zigzag()