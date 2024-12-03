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
    # 1. 初始化
    num_levels = 2 ** bits
    num_channels = image.shape[2]  # 获取通道数

    # 为每个通道创建聚类中心
    clusters = np.zeros((num_channels, num_levels))
    qImage = np.zeros_like(image, dtype=int)

    # 对每个通道分别进行Lloyd-Max量化
    for channel in range(num_channels):
        # 初始化当前通道的聚类中心（均匀分布）
        clusters[channel] = np.linspace(0, 255, num_levels)

        # 获取当前通道的像素值
        pixels = image[:, :, channel].flatten()

        # 2. 迭代优化
        while True:
            old_clusters = clusters[channel].copy()

            # 2.1 分配阶段：将像素分配到最近的聚类中心
            distances = np.abs(pixels[:, np.newaxis] - clusters[channel])
            labels = np.argmin(distances, axis=1)

            # 2.2 更新阶段：更新聚类中心
            for i in range(num_levels):
                if np.sum(labels == i) > 0:  # 非空聚类
                    clusters[channel, i] = np.mean(pixels[labels == i])
                else:  # 处理空聚类
                    max_cluster_idx = np.argmax(np.bincount(labels))
                    max_cluster_pixels = pixels[labels == max_cluster_idx]
                    clusters[channel, i] = np.mean(max_cluster_pixels[:len(max_cluster_pixels) // 2])
                    clusters[channel, max_cluster_idx] = np.mean(max_cluster_pixels[len(max_cluster_pixels) // 2:])

            # 2.3 检查收敛条件
            if np.max(np.abs(clusters[channel] - old_clusters)) < epsilon:
                break

        # 3. 量化当前通道的图像
        distances = np.abs(image[:, :, channel][:, :, np.newaxis] - clusters[channel])
        qImage[:, :, channel] = np.argmin(distances, axis=2)

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

    height, width, channels = qImage.shape
    image = np.zeros((height, width, channels))

    # 对每个通道分别进行重建
    for c in range(channels):
        # 使用对应通道的聚类中心进行重建
        image[:, :, c] = clusters[c, qImage[:, :, c]]

    image = np.clip(image, 0, 255)
    # YOUR CODE ENDS HERE
    return image

def vector_quantizer(image:np.array, bits: int, epsilon: float, bsize: int):
    """
    Perform vector quantization on an image using block-based processing.

    Parameters:
        image (ndarray): Input image (grayscale or multi-channel).
        bits (int): Number of quantization bits.
        epsilon (float): Convergence threshold.
        bsize (int): Block size.

    Returns:
        clusters (ndarray): Final codebook (quantization levels).
        Temp_clusters (list of ndarray): Codebooks at each iteration.
    """
    # NOTE: check the KMeans function of scikit-learn (sklearn.cluster.KMeans)
    # YOUR CODE STARTS HERE
    height, width, channels = image.shape
    n_clusters = 2 ** bits

    h_blocks = height // bsize
    w_blocks = width // bsize

    blocks = image.reshape(h_blocks, bsize, w_blocks, bsize, channels)
    blocks = blocks.transpose(0, 2, 1, 3, 4)
    vectors = blocks.reshape(-1, bsize * bsize * channels)

    kmeans = KMeans(n_clusters=n_clusters,
                    init='k-means++',
                    tol=epsilon,
                    n_init=10)

    kmeans.fit(vectors)
    clusters = kmeans.cluster_centers_
    clusters = clusters.reshape(n_clusters, bsize, bsize, channels)

    Temp_clusters = [clusters]  # 这里假设只记录最终的聚类中心

    # YOUR CODE ENDS HERE
    return clusters, Temp_clusters

def apply_vector_quantizer(image: np.array, clusters: np.array, bsize: int)->np.array:
    """
    Apply the vector quantization to an image using a pre-trained codebook (clusters).

    Parameters:
        image (ndarray): Input image (grayscale or multi-channel).
        clusters (ndarray): Codebook (quantization levels).
        bsize (int): Block size.

    Returns:
        qImage (ndarray): Quantized image with cluster indices.
    """
    # YOUR CODE STARTS HERE
    height, width, channels = image.shape
    h_blocks = height // bsize
    w_blocks = width // bsize

    # 将图像分成blocks
    blocks = image.reshape(h_blocks, bsize, w_blocks, bsize, channels)
    blocks = blocks.transpose(0, 2, 1, 3, 4)
    vectors = blocks.reshape(-1, bsize * bsize * channels)

    # 初始化量化图像
    qImage = np.zeros((h_blocks, w_blocks), dtype=int)

    # 对每个block找到最近的聚类中心
    for i, vector in enumerate(vectors):
        distances = np.linalg.norm(clusters.reshape(clusters.shape[0], -1) - vector, axis=1)
        qImage[i // w_blocks, i % w_blocks] = np.argmin(distances)
    # YOUR CODE ENDS HERE
    
    return qImage

def inv_vector_quantizer(qImage: np.array, clusters: np.array, block_size: int):
    """
    Reconstruct an image from a quantized image using a pre-trained codebook (clusters).

    Parameters:
        qImage (ndarray): Quantized image (containing cluster indices).
        clusters (ndarray): Codebook (quantization levels).
        block_size (int): Size of each block.

    Returns:
        image (ndarray): Reconstructed image.
    """
    # YOUR CODE STARTS HERE
    h_blocks, w_blocks = qImage.shape
    n_clusters, bsize, _, channels = clusters.shape

    # 初始化重建图像
    image = np.zeros((h_blocks * block_size, w_blocks * block_size, channels))

    # 对每个block进行重建
    for i in range(h_blocks):
        for j in range(w_blocks):
            cluster_idx = qImage[i, j]
            block = clusters[cluster_idx]
            image[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size, :] = block
    # YOUR CODE ENDS HERE

    return image