import numpy as np
from scipy.fft import dct, idct

class DiscreteCosineTransform:

    def __init__(self, norm='ortho'):
        self.norm = norm

    def transform(self, patched_img: np.array):


        # YOUR CODE STARTS HERE
        transformed = dct(patched_img, axis=-1, norm=self.norm)
        transformed = dct(transformed, axis=-2, norm=self.norm)
        # YOUR CODE ENDS HERE
        return transformed
    
    def inverse_transform(self, transformed: np.array):

        # YOUR CODE STARTS HERE
        patched_img = idct(transformed, axis=-1, norm=self.norm)
        patched_img = idct(patched_img, axis=-2, norm=self.norm)
        # YOUR CODE ENDS HERE
        return patched_img