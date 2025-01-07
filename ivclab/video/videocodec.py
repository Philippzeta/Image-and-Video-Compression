import numpy as np
from ivclab.entropy import ZeroRunCoder
from ivclab.image import IntraCodec
from ivclab.entropy import HuffmanCoder, stats_marg
from ivclab.signal import rgb2ycbcr, ycbcr2rgb
from ivclab.video import MotionCompensator

class VideoCodec:

    def __init__(self, 
                 quantization_scale = 1.0,
                 bounds = (-1000, 4000),
                 end_of_block = 4000,
                 block_shape = (8,8),
                 search_range = 4
                 ):
        
        self.quantization_scale = quantization_scale
        self.bounds = bounds
        self.end_of_block = end_of_block
        self.block_shape = block_shape
        self.search_range = search_range

        self.intra_codec = IntraCodec(quantization_scale=quantization_scale, bounds=bounds, end_of_block=end_of_block, block_shape=block_shape)
        self.motion_comp = MotionCompensator(search_range=search_range)
        self.motion_huffman = HuffmanCoder(lower_bound=((2*search_range + 1)**2 - 1)//2)

        self.encoding_first = True
        self.decoding_first = True
        self.prev_frame = None
        self.current_frame = None

    def encode(self, frame):
        if self.encoding_first:
            self.encoding_first = False
            # YOUR CODE STARTS HERE
            pass
            # YOUR CODE ENDS HERE
        else:
            # YOUR CODE STARTS HERE
            pass
            # YOUR CODE ENDS HERE
        return bitstream 
    
    def decode(self, bitstream):
        if self.decoding_first:
            self.decoding_first = False
            # YOUR CODE STARTS HERE
            pass
            # YOUR CODE ENDS HERE
        else:
            # YOUR CODE STARTS HERE
            pass
            # YOUR CODE ENDS HERE
        return frame
    

    

