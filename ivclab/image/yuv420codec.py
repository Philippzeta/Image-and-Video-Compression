import numpy as np
from scipy.signal import decimate, resample

def rgb2ycbcr(image: np.array):

    output_image = np.zeros_like(image)
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.169 * R - 0.331 * G + 0.5 * B
    Cr = 0.5 * R - 0.419 * G - 0.081 * B

    output_image[:, :, 0] = Y
    output_image[:, :, 1] = Cb
    output_image[:, :, 2] = Cr
    return output_image

def ycbcr2rgb(image: np.array):

    output_image = np.zeros_like(image)
    Y = image[:, :, 0]
    Cb = image[:, :, 1]
    Cr = image[:, :, 2]

    R = Y + 1.402 * Cr
    G = Y - 0.344 * Cb - 0.714 * Cr
    B = Y + 1.772 * Cb

    output_image[:, :, 0] = np.clip(R, 0, 255)
    output_image[:, :, 1] = np.clip(G, 0, 255)
    output_image[:, :, 2] = np.clip(B, 0, 255)
    return output_image
def yuv420compression(image: np.ndarray):

    # Cast image to floating point
    image = image * 1.0

    # YOUR CODE STARTS HERE
    #problem
    original_height, original_width, _ = image.shape

    # Step 1: Convert an image from RGB to YCbCr
    ycbcr_image = rgb2ycbcr(image)

    # Step 2: Compress the image
    # A. Pad the image with 4 pixels symmetric pixels on each side
    padded_image = np.pad(ycbcr_image, ((4, 4), (4, 4), (0, 0)), mode='symmetric')

    # B. Downsample only Cb and Cr channels with prefiltering
    Y = padded_image[:, :, 0]
    Cb = padded_image[:, :, 1]
    Cr = padded_image[:, :, 2]

    Cb_downsampled = decimate(Cb, 2, axis=0)
    Cb_downsampled = decimate(Cb_downsampled, 2, axis=1)
    Cr_downsampled = decimate(Cr, 2, axis=0)
    Cr_downsampled = decimate(Cr_downsampled, 2, axis=1)

    # C. Crop the image 2 pixels from each side to get rid of padding

    Cb_cropped = Cb_downsampled[2:-2, 2:-2]
    Cr_cropped = Cr_downsampled[2:-2, 2:-2]

    # Step 3: Apply rounding to Y, Cb and Cr channels

    Cb_cropped = np.round(Cb_cropped)
    Cr_cropped = np.round(Cr_cropped)
    # Step 4: Decompress the image
    # A. Pad the image with 2 pixels symmetric pixels on each side

    Cb_padded = np.pad(Cb_cropped, ((2, 2), (2, 2)), mode='symmetric')
    Cr_padded = np.pad(Cr_cropped, ((2, 2), (2, 2)), mode='symmetric')

    # B. Upsample Cb and Cr channels
    Cb_upsampled = resample(Cb_padded, Y.shape[0], axis=0)
    Cb_upsampled = resample(Cb_upsampled, Y.shape[1], axis=1)
    Cr_upsampled = resample(Cr_padded, Y.shape[0], axis=0)
    Cr_upsampled = resample(Cr_upsampled, Y.shape[1], axis=1)

    # C. Crop the image 4 pixels from each side to get rid of padding

    Cb_final = Cb_upsampled[4:-4, 4:-4]
    Cr_final = Cr_upsampled[4:-4, 4:-4]
    Y_final=Y[4:-4, 4:-4]


    # Combine Y, Cb, and Cr channels
    ycbcr_decompressed = np.zeros_like(image)
    ycbcr_decompressed[:, :, 0] = Y_final
    ycbcr_decompressed[:, :, 1] = Cb_final
    ycbcr_decompressed[:, :, 2] = Cr_final

    # Step 5: Convert YCbCr back to RGB
    output = ycbcr2rgb(ycbcr_decompressed)

    #YOUR CODE ENDS HERE

    # Cast output to integer again
    output = np.round(output).astype(np.uint8)
    return output