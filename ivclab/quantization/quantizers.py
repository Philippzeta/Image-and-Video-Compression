import numpy as np

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
    raise NotImplementedError()
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
    raise NotImplementedError()
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
    raise NotImplementedError()
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
    raise NotImplementedError()
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
    raise NotImplementedError()
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
    raise NotImplementedError()
    # YOUR CODE ENDS HERE
    
    return qImage

import numpy as np

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
    raise NotImplementedError()
    # YOUR CODE ENDS HERE

    return image