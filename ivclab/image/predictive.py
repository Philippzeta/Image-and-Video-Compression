import numpy as np
from scipy.signal import decimate
from ivclab.signal import rgb2ycbcr, ycbcr2rgb

def single_pixel_predictor(image):
    """
    Creates a residual image after a single pixel predictor for overlapping 
    pixel pairs. The right pixel is predicted from the left pixel with the formula
    R_pred = L * a1 where a1=1. This function returns the residual R - R_pred. For
    the first pixels of each row who don't have a left neighbor, it copies the values
    from the original image instead of making a prediction

    image: np.array of shape [H, W, C]

    returns 
        residual_image: np.array of shape [H, W, C]
    """
    # Convert image to floating points
    image = image * 1.0

    a1 = 1.0

    # Create residual image
    residual_image = np.zeros_like(image)

    # YOUR CODE STARTS HERE
    # Iterate through each row and channel
    for row in range(image.shape[0]):
        for channel in range(image.shape[2]):
            # For the first pixel in each row, copy the original value
            residual_image[row, 0, channel] = image[row, 0, channel]
            # For other pixels, calculate the prediction residual
            for col in range(1, image.shape[1]):
                predicted_value = a1 * image[row, col - 1, channel]
                residual_image[row, col, channel] = image[row, col, channel] - predicted_value
    # YOUR CODE ENDS HERE

    residual_image = np.round(np.clip(residual_image, -255, 255))

    return residual_image

def _predict_from_neighbors(original, coefficients):
    """
    Helper function for the three pixel predictor. Here is the main computation:

    prediction(current) = coefficients * reconstruction(previous)
    error(current) = round(original(current) - prediction(current))
    reconstruction(current) = prediction(current) + error(current)

    We need to create two arrays, reconstruction and residual_error. They are already
    initialized such that the top row and the leftmost column of the original image
    is copied to them.
    
    It applies this over all pixels from top-left to bottom-right in order.

    Hint: Start from the second index in "for loops" of both directions

    original: np.array of shape [H, W, C]
    reconstruction: np.array of shape [H, W, C]
    residual_error: np.array of shape [H, W, C]
    coefficients: list of 3 floating point numbers (see lab slides for what they represent)

    returns 
        residual_error: np.array of shape [H, W, C]
    """
    H, W, C = original.shape

    reconstruction = np.zeros_like(original)
    reconstruction[0,:,:] = original[0,:,:]
    reconstruction[:,0,:] = original[:,0,:]

    residual_error = np.copy(reconstruction)

    # YOUR CODE STARTS HERE

    for row in range(1, H):
        for col in range(1, W):
            for channel in range(C):
                # Previous values (left, above, and top-left)
                left = reconstruction[row, col - 1, channel]
                above = reconstruction[row - 1, col, channel]
                top_left = reconstruction[row - 1, col - 1, channel]

                # Calculate the prediction based on neighbors cannot be realized
                predicted_value = (coefficients[0] * left + coefficients[1] * top_left + coefficients[2] * above)

                # Calculate the residual error
                residual_error[row, col, channel] = round(original[row, col, channel] - predicted_value)

                # Update reconstruction using the residual
                reconstruction[row, col, channel] = predicted_value + residual_error[row, col, channel]

    # YOUR CODE ENDS HERE

    return residual_error

def three_pixels_predictor(image, subsample_color_channels=False):
    """
    Creates a residual image after a three pixels predictor.

    1. Convert the input image to YCbCr color space
    2. If subsample_color_channels, then subsample the Cb and Cr channels
        by 2, similar to the yuv420codec (use scipy.signal.decimate)
    3. Apply three pixel prediction with the given coefficients for Y and CbCr channels.
        You must use _predict_from_neighbors helper function
    4. Return the residual error images

    image: np.array of shape [H, W, C]

    returns 
        residual_image_Y: np.array of shape [H, W, 1]
        residual_image_CbCr: np.array of shape [H, W, 2] (or [H // 2, W // 2, 2] if subsampled)
    """
    # Convert image to floating points
    image = image * 1.0

    coefficients_Y = [7/8, -4/8, 5/8]
    coefficients_CbCr = [3/8, -2/8, 7/8]

    # YOUR CODE STARTS HERE
    # Step 1: Convert the input image to YCbCr color space
    ycbcr_image = rgb2ycbcr(image)

    # Step 2: Split the channels into Y, Cb, and Cr
    Y_channel = ycbcr_image[:, :, 0:1]  # Shape [H, W, 1]
    Cb_channel = ycbcr_image[:, :, 1:2]  # Shape [H, W, 1]
    Cr_channel = ycbcr_image[:, :, 2:3]  # Shape [H, W, 1]

    # Step 3: subsample Cb and Cr channels
    if subsample_color_channels:
        Cb_channel = decimate(Cb_channel.squeeze(), 2, axis=0)
        Cb_channel = decimate(Cb_channel, 2, axis=1)[..., np.newaxis]
        Cr_channel = decimate(Cr_channel.squeeze(), 2, axis=0)
        Cr_channel = decimate(Cr_channel, 2, axis=1)[..., np.newaxis]

    # Predict Y channel
    residual_image_Y = _predict_from_neighbors(Y_channel, coefficients_Y)

    # Predict Cb and Cr channels
    residual_image_Cb = _predict_from_neighbors(Cb_channel, coefficients_CbCr)
    residual_image_Cr = _predict_from_neighbors(Cr_channel, coefficients_CbCr)

    # Stack Cb and Cr together if not subsampled, else keep separate
    if subsample_color_channels:
        residual_image_CbCr = np.concatenate([residual_image_Cb, residual_image_Cr], axis=-1)
    else:
        residual_image_CbCr = np.stack([residual_image_Cb, residual_image_Cr], axis=-1)

    # YOUR CODE ENDS HERE

    residual_image_Y = np.round(np.clip(residual_image_Y, -255, 255)).astype(np.int32)
    residual_image_CbCr = np.round(np.clip(residual_image_CbCr, -255, 255)).astype(np.int32)

    return residual_image_Y, residual_image_CbCr