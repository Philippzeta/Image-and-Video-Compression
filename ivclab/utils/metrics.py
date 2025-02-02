import numpy as np

def calc_mse(orig: np.array, rec: np.array):

    # YOUR CODE STARTS HERE
    #can not just use np.sum((orig - rec) ** 2)
    mse = np.sum((orig.astype(np.float32) - rec.astype(np.float32)) ** 2) / (orig.shape[0] * orig.shape[1] * orig.shape[2])
    # YOUR CODE ENDS HERE
    return mse

def calc_psnr(orig: np.array, rec:np.array, maxval=255):

    # YOUR CODE STARTS HERE
    mse = calc_mse(orig, rec)
    if mse == 0:
        return float('inf')
    #or maxval here 255 is 8bit image
    psnr = 10 * np.log10((maxval ** 2) / mse)
    # YOUR CODE ENDS HERE
    return psnr