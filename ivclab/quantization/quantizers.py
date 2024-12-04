import numpy as np
from sklearn.cluster import KMeans
def uniquant(image: np.array, bits: np.array)->np.array:
    """
    Function corresponding to E(3-1a) in the handout. 
    The function should operate in the 8-bit range of [0, 255] pixel intensities and output the 
    quantized image, consisting of quantization indices of the representatives using scalar 
    uniform quantization.

    image: np.array of shape [H, W, C]
    bits: np.array of shape [H, W, C]

    returns 
        qImage: np.array, quantized
    """
    # YOUR CODE STARTS HERE
    normalized = image / 255.0
    levels = 2 ** bits
    step_size = 1.0 / levels
    qImage = np.floor(normalized / step_size)
    qImage = np.clip(qImage, 0, levels - 1)
    # YOUR CODE ENDS HERE
    return qImage

def inv_uniquant(qImage: np.array, bits: int)->np.array:
    """
    Inverse uniform quantization function.

    Parameters:
        qImage (ndarray): Quantized image as a NumPy array.
        bits (int): Number of bits available for representatives.

    Returns:
        ndarray: Reconstructed image.
    """
    # YOUR CODE STARTS HERE
    levels = 2 ** bits
    step_size = 1.0 / levels
    normalized = (qImage + 0.5) * step_size
    image = np.clip(normalized * 255.0, 0, 255)
    # YOUR CODE ENDS HERE
    return image

def lloyd_max(image: np.array, bits: int, epsilon: float):
    """
    Lloyd-Max quantization algorithm for optimal quantization levels.

    Parameters:
        image (ndarray): Input image as a NumPy array.
        bits (int): Number of bits for quantization.
        epsilon (float): Convergence threshold.

    Returns:
        qImage (ndarray): Quantized image.
        clusters (ndarray): Optimal quantization levels (clusters).
    """
    # YOUR CODE STARTS HERE
    # 初始化
    M = 2 ** bits  # number of representatives
    partition_size = 256 / M
    rep = np.arange(partition_size / 2, 256, partition_size)
    codeBook = np.zeros((M, 3))
    codeBook[:, 0] = rep
    d = [9999]  # stored distortion
    sz = image.shape

    while True:
        d_quad = 0
        for channel in range(3):
            pixel_value = image[:, :, channel].flatten()

            # 计算距离并找到最近的代表值
            distances = np.abs(codeBook[:, 0, np.newaxis] - pixel_value)
            I = np.argmin(distances, axis=0)
            D = distances[I, np.arange(len(pixel_value))]

            # 更新codebook
            for kk in range(len(pixel_value)):
                index = I[kk]
                codeBook[index, 1] += pixel_value[kk]
                codeBook[index, 2] += 1

            d_quad += np.sum(D ** 2)

        # 更新和重置codebook
        index_zeroUpdate = np.where(codeBook[:, 2] == 0)[0]
        if len(index_zeroUpdate) > 0:
            # 处理空单元格
            for k in index_zeroUpdate:
                index_max = np.argmax(codeBook[:, 2])
                count_max = codeBook[index_max, 2]
                codeBook[index_max, 2] = np.ceil(count_max / 2)
                codeBook[k, 2] = count_max - codeBook[index_max, 2]
                codeBook[k, 1] = codeBook[index_max, 1] + 1

        # 更新代表值
        mask = codeBook[:, 2] != 0
        codeBook[mask, 0] = codeBook[mask, 1] / codeBook[mask, 2]
        codeBook[:, 1:] = 0

        # 检查停止条件
        d.append(d_quad)
        J = abs(d[-1] - d[-2]) / d[-1]
        if J < epsilon:
            break

    # 量化图像
    qImage = np.zeros(sz, dtype=int)
    for channel in range(3):
        pixel_value = image[:, :, channel]
        distances = np.abs(codeBook[:, 0, np.newaxis] - pixel_value.flatten())
        I = np.argmin(distances, axis=0)
        qImage[:, :, channel] = I.reshape(sz[:2])

    clusters = codeBook[:, 0]
    # YOUR CODE ENDS HERE

    return qImage, clusters



def inv_lloyd_max(qImage: np.array, clusters: np.array)->np.array:
    """
    Inverse Lloyd-Max quantization to reconstruct the image.

    Parameters:
        qImage (ndarray): Quantized image (indices of clusters).
        clusters (ndarray): Array of cluster center values.

    Returns:
        ndarray: Reconstructed image.
    """
    
    # YOUR CODE STARTS HERE
    numRep = len(clusters)
    sz = qImage.shape
    image = np.zeros(sz)

    for channel in range(3):
        quantized = np.zeros(sz[0] * sz[1])
        for i in range(numRep):
            pixel_value = qImage[:, :, channel].flatten()
            index = np.where(pixel_value == i)[0]
            quantized[index] = clusters[i]
        image[:, :, channel] = quantized.reshape(sz[:2])
    # YOUR CODE ENDS HERE
    return image
"""
def vector_quantizer(image:np.array, bits: int, epsilon: float, bsize: int):
    
    Perform vector quantization on an image using block-based processing.

    Parameters:
        image (ndarray): Input image (grayscale or multi-channel).
        bits (int): Number of quantization bits.
        epsilon (float): Convergence threshold.
        bsize (int): Block size.

    Returns:
        clusters (ndarray): Final codebook (quantization levels).
        Temp_clusters (list of ndarray): Codebooks at each iteration.
    
    # NOTE: check the KMeans function of scikit-learn (sklearn.cluster.KMeans)
    # YOUR CODE STARTS HERE
    # 1. 确保图像数据类型为float64，与MATLAB保持一致
    image = image.astype(np.float64)

    height, width, channels = image.shape
    h_blocks = height // bsize
    w_blocks = width // bsize

    # 2. 将图像分成块，保持与MATLAB相同的处理顺序
    blocks = []
    for i in range(0, height - bsize + 1, bsize):
        for j in range(0, width - bsize + 1, bsize):
            block = image[i:i + bsize, j:j + bsize, :].copy()
            blocks.append(block.reshape(-1))
    blocks = np.array(blocks)

    # 3. 使用KMeans训练codebook
    n_clusters = 2 ** bits
    kmeans = KMeans(
        n_clusters=n_clusters,
        tol=epsilon,
        init='random',  # 使用random初始化
        n_init=1,  # 单次运行
        max_iter=300,
        random_state=42
    )

    # 4. 训练并获取聚类中心
    kmeans.fit(blocks)
    clusters = kmeans.cluster_centers_

    # 5. 按照能量排序聚类中心
    energy = np.sum(clusters ** 2, axis=1)
    sorted_idx = np.argsort(energy)
    clusters = clusters[sorted_idx]

    Temp_clusters = [clusters]
    # YOUR CODE ENDS HERE
    return clusters, Temp_clusters

    
def apply_vector_quantizer(image: np.array, clusters: np.array, bsize: int)->np.array:
    
    Apply the vector quantization to an image using a pre-trained codebook (clusters).

    Parameters:
        image (ndarray): Input image (grayscale or multi-channel).
        clusters (ndarray): Codebook (quantization levels).
        bsize (int): Block size.

    Returns:
        qImage (ndarray): Quantized image with cluster indices.
    
    # YOUR CODE STARTS HERE
    # 1. 确保图像数据类型一致
    image = image.astype(np.float64)

    height, width, channels = image.shape
    h_blocks = height // bsize
    w_blocks = width // bsize

    # 2. 将图像分成块，保持相同的处理顺序
    blocks = []
    for i in range(0, height - bsize + 1, bsize):
        for j in range(0, width - bsize + 1, bsize):
            block = image[i:i + bsize, j:j + bsize, :].copy()
            blocks.append(block.reshape(-1))
    blocks = np.array(blocks)

    # 3. 计算每个块到聚类中心的距离
    distances = np.zeros((len(blocks), len(clusters)))
    for i, cluster in enumerate(clusters):
        diff = blocks - cluster
        distances[:, i] = np.sum(diff ** 2, axis=1)

    # 4. 找到最近的聚类中心
    indices = np.argmin(distances, axis=1)
    qImage = indices.reshape(h_blocks, w_blocks)

    # 保存维度信息
    global _block_dimensions
    _block_dimensions = (h_blocks, w_blocks)
    # YOUR CODE ENDS HERE
    
    return qImage

def inv_vector_quantizer(qImage: np.array, clusters: np.array, block_size: int):
    
    Reconstruct an image from a quantized image using a pre-trained codebook (clusters).

    Parameters:
        qImage (ndarray): Quantized image (containing cluster indices).
        clusters (ndarray): Codebook (quantization levels).
        block_size (int): Size of each block.

    Returns:
        image (ndarray): Reconstructed image.
    
    # YOUR CODE STARTS HERE
    # 1. 处理量化索引
    if qImage.ndim == 1:
        global _block_dimensions
        h_blocks, w_blocks = _block_dimensions
        qImage = qImage.reshape(h_blocks, w_blocks)
    else:
        h_blocks, w_blocks = qImage.shape

    # 2. 计算图像尺寸
    height = h_blocks * block_size
    width = w_blocks * block_size
    channels = clusters.shape[1] // (block_size * block_size)

    # 3. 重建图像
    image = np.zeros((height, width, channels), dtype=np.float64)
    for i in range(h_blocks):
        for j in range(w_blocks):
            idx = qImage[i, j]
            block = clusters[idx].reshape(block_size, block_size, channels)
            image[i * block_size:(i + 1) * block_size,
            j * block_size:(j + 1) * block_size, :] = block

    # 4. 确保像素值在正确范围内
    image = np.clip(image, 0, 255)

    # YOUR CODE ENDS HERE
    return image
"""


def vector_quantizer(image: np.array, bits: int, epsilon: float, bsize: int):
    """
    为每个通道分别训练码本
    Parameters:
        image: 输入图像 [H, W, C]
        bits: 量化位数
        epsilon: 收敛阈值
        bsize: 块大小
    Returns:
        clusters: 每个通道的VQ Table [C, n_clusters, block_size^2]
        Temp_clusters: 训练过程中的码本
    """
    height, width, channels = image.shape
    n_clusters = 2 ** bits

    # 为每个通道准备单独的码本
    all_clusters = []
    all_temp_clusters = []

    # 对每个通道分别训练
    for channel in range(channels):
        # 准备当前通道的训练数据
        channel_blocks = []
        for i in range(0, height - bsize + 1, bsize):
            for j in range(0, width - bsize + 1, bsize):
                block = image[i:i + bsize, j:j + bsize, channel].reshape(-1)
                channel_blocks.append(block)
        channel_blocks = np.array(channel_blocks)

        # 使用KMeans训练当前通道的码本
        kmeans = KMeans(
            n_clusters=n_clusters,
            tol=epsilon,
            random_state=42,
            n_init=1
        )
        kmeans.fit(channel_blocks)

        # 保存当前通道的码本
        all_clusters.append(kmeans.cluster_centers_)
        all_temp_clusters.append([kmeans.cluster_centers_])

    # 将所有通道的码本组合
    clusters = np.array(all_clusters)  # shape: [C, n_clusters, block_size^2]
    Temp_clusters = all_temp_clusters  # list of length C

    return clusters, Temp_clusters


def apply_vector_quantizer(image: np.array, clusters: np.array, bsize: int) -> np.array:
    """
    使用每个通道的码本进行量化
    Parameters:
        image: 输入图像 [H, W, C]
        clusters: 每个通道的码本 [C, n_clusters, block_size^2]
        bsize: 块大小
    Returns:
        qImage: 量化后的图像索引 [H/bsize, W/bsize, C]
    """
    height, width, channels = image.shape
    num_row = height // bsize
    num_col = width // bsize

    # 初始化输出
    qImage = np.zeros((num_row, num_col, channels), dtype=np.int32)

    # 对每个通道分别处理
    for channel in range(channels):
        # 提取当前通道的块
        blocks = []
        for i in range(0, height - bsize + 1, bsize):
            for j in range(0, width - bsize + 1, bsize):
                block = image[i:i + bsize, j:j + bsize, channel].reshape(-1)
                blocks.append(block)
        blocks = np.array(blocks)

        # 使用当前通道的码本
        channel_clusters = clusters[channel]

        # 找到最近的码字
        distances = np.sum((blocks[:, np.newaxis, :] - channel_clusters) ** 2, axis=2)
        indices = np.argmin(distances, axis=1)

        # 重塑为所需维度
        qImage[:, :, channel] = indices.reshape(num_row, num_col)

    return qImage


def inv_vector_quantizer(qImage: np.array, clusters: np.array, block_size: int) -> np.array:
    """
    使用每个通道的码本重建图像
    Parameters:
        qImage: 量化索引图像 [H/block_size, W/block_size, C]
        clusters: 每个通道的码本 [C, n_clusters, block_size^2]
        block_size: 块大小
    Returns:
        image: 重建后的图像 [H, W, C]
    """
    height, width, channels = qImage.shape
    original_height = height * block_size
    original_width = width * block_size

    # 初始化输出图像
    image = np.zeros((original_height, original_width, channels))

    # 对每个通道分别处理
    for channel in range(channels):
        # 获取当前通道的量化索引和码本
        channel_qimage = qImage[:, :, channel]
        channel_clusters = clusters[channel]

        # 重建当前通道
        for i in range(height):
            for j in range(width):
                cluster_idx = channel_qimage[i, j]
                block = channel_clusters[cluster_idx].reshape(block_size, block_size)
                image[i * block_size:(i + 1) * block_size,
                j * block_size:(j + 1) * block_size,
                channel] = block

    return image