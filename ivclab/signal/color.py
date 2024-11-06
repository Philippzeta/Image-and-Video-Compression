import numpy as np

def rgb2gray(image: np.array):
    """
    Computes the grayscale version of the image. 

    image: np.array of shape [H, W, C]

    returns 
        output_image: np.array of shape [H, W, 1]
    """
    output_image = np.mean(image, axis=-1, keepdims=True)
    return output_image

def rgb2ycbcr(image: np.array):
    """
    Converts an RGB image to its YCbCr version. 

    image: np.array of shape [H, W, 3]

    returns 
        output_image: np.array of shape [H, W, 3]
    """
    output_image = np.zeros_like(image)
    # YOUR CODE STARTS HERE
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.169 * R - 0.331 * G + 0.5 * B
    Cr = 0.5 * R - 0.419 * G - 0.081 * B

    output_image[:, :, 0] = Y
    output_image[:, :, 1] = Cb
    output_image[:, :, 2] = Cr
    # YOUR CODE ENDS HERE
    return output_image

def ycbcr2rgb(image: np.array):
    """
    Converts an YCbCr image to its RGB version. 

    image: np.array of shape [H, W, 3]

    returns 
        output_image: np.array of shape [H, W, 3]
    """
    output_image = np.zeros_like(image)
    # YOUR CODE STARTS HERE
    Y = image[:, :, 0]
    Cb = image[:, :, 1]
    Cr = image[:, :, 2]

    R = Y + 1.402 * Cr
    G = Y - 0.344 * Cb - 0.714 * Cr
    B = Y + 1.772 * Cb

    output_image[:, :, 0] = np.clip(R, 0, 255)
    output_image[:, :, 1] = np.clip(G, 0, 255)
    output_image[:, :, 2] = np.clip(B, 0, 255)

    # YOUR CODE ENDS HERE
    return output_image