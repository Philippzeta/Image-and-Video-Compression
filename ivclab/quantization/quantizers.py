import numpy as np
from sklearn.cluster import KMeans
def uniquant(image: np.array, bits: np.array)->np.array:

    # YOUR CODE STARTS HERE
    normalized = image / 255.0
    levels = 2 ** bits
    step_size = 1.0 / levels
    qImage = np.floor(normalized / step_size)
    qImage = np.clip(qImage, 0, levels - 1)
    # YOUR CODE ENDS HERE
    return qImage

def inv_uniquant(qImage: np.array, bits: int)->np.array:

    # YOUR CODE STARTS HERE
    levels = 2 ** bits
    step_size = 1.0 / levels
    normalized = (qImage + 0.5) * step_size
    image = np.clip(normalized * 255.0, 0, 255)
    # YOUR CODE ENDS HERE
    return image

def lloyd_max(image: np.array, bits: int, epsilon: float):

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


            distances = np.abs(codeBook[:, 0, np.newaxis] - pixel_value)
            I = np.argmin(distances, axis=0)
            D = distances[I, np.arange(len(pixel_value))]

            # codebook
            for kk in range(len(pixel_value)):
                index = I[kk]
                codeBook[index, 1] += pixel_value[kk]
                codeBook[index, 2] += 1

            d_quad += np.sum(D ** 2)

        # codebook
        index_zeroUpdate = np.where(codeBook[:, 2] == 0)[0]
        if len(index_zeroUpdate) > 0:

            for k in index_zeroUpdate:
                index_max = np.argmax(codeBook[:, 2])
                count_max = codeBook[index_max, 2]
                codeBook[index_max, 2] = np.ceil(count_max / 2)
                codeBook[k, 2] = count_max - codeBook[index_max, 2]
                codeBook[k, 1] = codeBook[index_max, 1] + 1

        mask = codeBook[:, 2] != 0
        codeBook[mask, 0] = codeBook[mask, 1] / codeBook[mask, 2]
        codeBook[:, 1:] = 0

        d.append(d_quad)
        J = abs(d[-1] - d[-2]) / d[-1]
        if J < epsilon:
            break

    # QU IMAGE
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

def vector_quantizer(image: np.array, bits: int, epsilon: float, bsize: int):

    height, width, channels = image.shape
    n_clusters = 2 ** bits

    # Separate codebooks for each channel
    all_clusters = []
    all_temp_clusters = []

    # Train each channel separately
    for channel in range(channels):
        # Prepare training data for the current channel
        channel_blocks = []
        for i in range(0, height - bsize + 1, bsize):
            for j in range(0, width - bsize + 1, bsize):
                block = image[i:i + bsize, j:j + bsize, channel].reshape(-1)
                channel_blocks.append(block)
        channel_blocks = np.array(channel_blocks)

        # Using KMeans to train the codebook for the current channel
        kmeans = KMeans(
            n_clusters=n_clusters,
            tol=epsilon,
            random_state=42,
            n_init=1
        )
        kmeans.fit(channel_blocks)

        # Save the codebook of the current channel
        all_clusters.append(kmeans.cluster_centers_)
        all_temp_clusters.append([kmeans.cluster_centers_])

    # Combine codebooks from all channels
    clusters = np.array(all_clusters)  # shape: [C, n_clusters, block_size^2]
    Temp_clusters = all_temp_clusters  # list of length C

    return clusters, Temp_clusters


def apply_vector_quantizer(image: np.array, clusters: np.array, bsize: int) -> np.array:

    height, width, channels = image.shape
    num_row = height // bsize
    num_col = width // bsize

    # Initialization Output
    qImage = np.zeros((num_row, num_col, channels), dtype=np.int32)

    # Process each channel separately
    for channel in range(channels):
        # Extract the current channel's blocks
        blocks = []
        for i in range(0, height - bsize + 1, bsize):
            for j in range(0, width - bsize + 1, bsize):
                block = image[i:i + bsize, j:j + bsize, channel].reshape(-1)
                blocks.append(block)
        blocks = np.array(blocks)

        # Use the codebook of the current channel
        channel_clusters = clusters[channel]

        # Find the nearest code word
        distances = np.sum((blocks[:, np.newaxis, :] - channel_clusters) ** 2, axis=2)
        indices = np.argmin(distances, axis=1)

        # Reshape to desired dimensions
        qImage[:, :, channel] = indices.reshape(num_row, num_col)

    return qImage


def inv_vector_quantizer(qImage: np.array, clusters: np.array, block_size: int) -> np.array:

    height, width, channels = qImage.shape
    original_height = height * block_size
    original_width = width * block_size

    # Initialize the output image
    image = np.zeros((original_height, original_width, channels))

    # Process each channel separately
    for channel in range(channels):
        # Get the quantization index and codebook of the current channel
        channel_qimage = qImage[:, :, channel]
        channel_clusters = clusters[channel]

        # Rebuild current channel
        for i in range(height):
            for j in range(width):
                cluster_idx = channel_qimage[i, j]
                block = channel_clusters[cluster_idx].reshape(block_size, block_size)
                image[i * block_size:(i + 1) * block_size,
                j * block_size:(j + 1) * block_size,
                channel] = block

    return image