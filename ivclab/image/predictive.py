import numpy as np
from ivclab.signal import rgb2ycbcr, ycbcr2rgb
from scipy.signal import decimate, resample

def single_pixel_predictor(image):

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


    # Stack Cb and Cr together
    residual_image_CbCr = np.concatenate([residual_image_Cb, residual_image_Cr], axis=2)  # 变成 [H, W, 2]
    # Stack Cb and Cr together if not subsampled, else keep separate
    #if subsample_color_channels:
        #residual_image_CbCr = np.concatenate([residual_image_Cb, residual_image_Cr], axis=-1)
    #else:
        #residual_image_CbCr = np.stack([residual_image_Cb, residual_image_Cr], axis=-1)

    # YOUR CODE ENDS HERE

    residual_image_Y = np.round(np.clip(residual_image_Y, -255, 255)).astype(np.int32)
    residual_image_CbCr = np.round(np.clip(residual_image_CbCr, -255, 255)).astype(np.int32)

    return residual_image_Y, residual_image_CbCr


###################################################################################################################
def _predict_from_neighborsnew(original, coefficients):

    original = original.astype(np.float32)
    H, W, C = original.shape

    reconstruction = np.zeros_like(original)
    reconstruction[0, :, :] = original[0, :, :]
    reconstruction[:, 0, :] = original[:, 0, :]

    residual_error = np.copy(reconstruction)


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


    return residual_error

def yuv420_three_pixel_residual(ycbcr_image: np.ndarray):
    """
    结合YUV420的采样和三像素预测来生成残差。

    Steps:
    1. Pad the image
    2. Downsample Cb and Cr channels
    3. Apply three-pixel prediction
    4. Return residuals

    Args:
        ycbcr_image: YCbCr格式的输入图像 [H, W, 3]
    Returns:
        residual_Y: Y通道的残差 [H, W, 1]
        residual_CbCr: CbCr通道的残差 [H//2, W//2, 2]
    """
    coefficients_Y = [7 / 8, -4 / 8, 5 / 8]
    coefficients_CbCr = [3 / 8, -2 / 8, 7 / 8]

    # Step 1: Pad the image
    padded_image = np.pad(ycbcr_image, ((4, 4), (4, 4), (0, 0)), mode='symmetric')

    # Step 2: Split and downsample Cb/Cr channels
    Y = padded_image[:, :, 0:1]
    Cb = decimate(padded_image[:, :, 1], 2, axis=0)
    Cb = decimate(Cb, 2, axis=1)[..., np.newaxis]
    Cr = decimate(padded_image[:, :, 2], 2, axis=0)
    Cr = decimate(Cr, 2, axis=1)[..., np.newaxis]

    # Step 3: Apply three-pixel prediction
    # For Y channel
    residual_Y = _predict_from_neighborsnew(Y, coefficients_Y)
    residual_Y = residual_Y[4:-4, 4:-4, :]  # Remove padding

    # For Cb and Cr channels
    residual_Cb = _predict_from_neighborsnew(Cb, coefficients_CbCr)
    residual_Cr = _predict_from_neighborsnew(Cr, coefficients_CbCr)

    # Combine Cb and Cr residuals
    residual_CbCr = np.concatenate([residual_Cb, residual_Cr], axis=2)

    return residual_Y, residual_CbCr


def reconstruct_ycbcr_from_residuals(decoded_residuals, original_shape):
    """
    从解码的残差重构YCbCr图像。

    Steps:
    1. Split residuals back into Y and CbCr
    2. Reconstruct using three-pixel prediction
    3. Pad for upsampling
    4. Upsample CbCr channels
    5. Combine channels

    Args:
        decoded_residuals: 解码后的残差数组
        original_shape: 原始图像形状 (H, W, 3)
    Returns:
        reconstructed_ycbcr: 重构的YCbCr图像 [H, W, 3]
    """
    H, W, _ = original_shape

    # 分离Y和CbCr残差
    Y_size = H * W  # Y通道大小
    residual_Y = decoded_residuals[:Y_size].reshape(H, W, 1)

    # CbCr残差使用实际大小 (260, 260, 2)
    CbCr_H, CbCr_W = 260, 260  # 从打印信息中得知的实际大小
    residual_CbCr = decoded_residuals[Y_size:].reshape(CbCr_H, CbCr_W, 2)

    # 初始化重构数组
    padded_Y = np.pad(np.zeros((H, W, 1)), ((4, 4), (4, 4), (0, 0)), mode='symmetric')
    padded_Cb = np.pad(np.zeros((CbCr_H, CbCr_W, 1)), ((2, 2), (2, 2), (0, 0)), mode='symmetric')
    padded_Cr = np.pad(np.zeros((CbCr_H, CbCr_W, 1)), ((2, 2), (2, 2), (0, 0)), mode='symmetric')

    # 复制第一行和第一列
    padded_Y[4:, 4, 0] = residual_Y[:, 0, 0]
    padded_Y[4, 4:, 0] = residual_Y[0, :, 0]
    padded_Cb[2:, 2, 0] = residual_CbCr[:, 0, 0]
    padded_Cb[2, 2:, 0] = residual_CbCr[0, :, 0]
    padded_Cr[2:, 2, 0] = residual_CbCr[:, 0, 1]
    padded_Cr[2, 2:, 0] = residual_CbCr[0, :, 1]

    # 重构Y通道
    coefficients_Y = [7 / 8, -4 / 8, 5 / 8]
    for i in range(1, H):
        for j in range(1, W):
            left = padded_Y[i + 4, j + 3, 0]
            top = padded_Y[i + 3, j + 4, 0]
            top_left = padded_Y[i + 3, j + 3, 0]
            pred = (coefficients_Y[0] * left +
                    coefficients_Y[1] * top_left +
                    coefficients_Y[2] * top)
            padded_Y[i + 4, j + 4, 0] = pred + residual_Y[i, j, 0]

    # 重构Cb和Cr通道
    coefficients_CbCr = [3 / 8, -2 / 8, 7 / 8]
    for i in range(1, CbCr_H):
        for j in range(1, CbCr_W):
            # Cb通道
            left = padded_Cb[i + 2, j + 1, 0]
            top = padded_Cb[i + 1, j + 2, 0]
            top_left = padded_Cb[i + 1, j + 1, 0]
            pred = (coefficients_CbCr[0] * left +
                    coefficients_CbCr[1] * top_left +
                    coefficients_CbCr[2] * top)
            padded_Cb[i + 2, j + 2, 0] = pred + residual_CbCr[i, j, 0]

            # Cr通道
            left = padded_Cr[i + 2, j + 1, 0]
            top = padded_Cr[i + 1, j + 2, 0]
            top_left = padded_Cr[i + 1, j + 1, 0]
            pred = (coefficients_CbCr[0] * left +
                    coefficients_CbCr[1] * top_left +
                    coefficients_CbCr[2] * top)
            padded_Cr[i + 2, j + 2, 0] = pred + residual_CbCr[i, j, 1]

    # 上采样CbCr通道到原始大小
    Cb_upsampled = resample(padded_Cb[2:-2, 2:-2, 0], H, axis=0)
    Cb_upsampled = resample(Cb_upsampled, W, axis=1)
    Cr_upsampled = resample(padded_Cr[2:-2, 2:-2, 0], H, axis=0)
    Cr_upsampled = resample(Cr_upsampled, W, axis=1)

    # 组合成最终的YCbCr图像
    reconstructed_ycbcr = np.zeros((H, W, 3))
    reconstructed_ycbcr[:, :, 0] = padded_Y[4:-4, 4:-4, 0]
    reconstructed_ycbcr[:, :, 1] = Cb_upsampled
    reconstructed_ycbcr[:, :, 2] = Cr_upsampled

    return reconstructed_ycbcr
