import numpy as np
from scipy.fft import dct, idct

class DiscreteCosineTransform:
    """
    A class to implement the forward and inverse transform of the DCT Type II
    """

    def __init__(self, norm='ortho'):
        self.norm = norm

    def transform(self, patched_img: np.array):
        """
        Implements the DCT Type II using scipy.fft.dct function by 
        running it sequentially on the last two axes. Make sure you use 
        the norm specified in the object initialization.

        patched_img: np.array of shape [H_patch, W_patch, C, H_window, W_window]

        returns:
            transformed_img: np.array of shape [H_patch, W_patch, C, H_window, W_window]
        """

        # YOUR CODE STARTS HERE
        transformed = dct(patched_img, axis=-1, norm=self.norm)
        transformed = dct(transformed, axis=-2, norm=self.norm)
        # YOUR CODE ENDS HERE
        return transformed
    
    def inverse_transform(self, transformed: np.array):
        """
        Implements the IDCT Type II using scipy.fft.idct function by 
        running it sequentially on the last two axes. Make sure you use 
        the norm specified in the object initialization.

        transformed: np.array of shape [H_patch, W_patch, C, H_window, W_window]

        returns:
            patched_img: np.array of shape [H_patch, W_patch, C, H_window, W_window]
        """
        # YOUR CODE STARTS HERE
        patched_img = idct(transformed, axis=-1, norm=self.norm)
        patched_img = idct(patched_img, axis=-2, norm=self.norm)
        # YOUR CODE ENDS HERE
        return patched_img