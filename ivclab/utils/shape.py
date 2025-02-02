import numpy as np
from einops import rearrange

class ZigZag:

    def __init__(self):
        self.zigzag_order = np.asarray([
            0, 1, 5, 6, 14, 15, 27, 28,
            2, 4, 7, 13, 16, 26, 29, 42,
            3, 8, 12, 17, 25, 30, 41, 43,
            9, 11, 18, 24, 31, 40, 44, 53,
            10, 19, 23, 32, 39, 45, 52, 54,
            20, 22, 33, 38, 46, 51, 55, 60,
            21, 34, 37, 47, 50, 56, 59, 61,
            35, 36, 48, 49, 57, 58, 62, 63
        ])

    def flatten(self, patched_img: np.array):
        flattened = rearrange(patched_img, 'h w c p0 p1 -> h w c (p0 p1)')
        shuffled = flattened[:,:,:,self.zigzag_order]
        return shuffled

    def unflatten(self, flat_img):
        unshuffled = np.zeros_like(flat_img)
        unshuffled[:,:,:,self.zigzag_order] = flat_img
        unflattened = rearrange(unshuffled, 'h w c (p0 p1) -> h w c p0 p1', p0=8, p1=8)
        return unflattened

class Patcher:

    def __init__(self, window_size=(8,8)):
        self.window_size = window_size

    def patch(self, img: np.array):

        return rearrange(img, '(h p0) (w p1) c -> h w c p0 p1', p0=self.window_size[0], p1=self.window_size[1])
    
    def unpatch(self, patched_img: np.array):

        return rearrange(patched_img, 'h w c p0 p1 -> (h p0) (w p1) c', p0=self.window_size[0], p1=self.window_size[1])
    