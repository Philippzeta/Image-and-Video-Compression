from ivclab import HuffmanCoder
from ivclab.utils import imread
from ivclab.entropy import stats_marg
from ivclab.image import three_pixels_predictor
import numpy as np


lena_img = imread(f'../../data/lena_small.tif')
residual_image_Y, residual_image_CbCr = three_pixels_predictor(lena_img, subsample_color_channels=False)
merged_residuals = np.concatenate([residual_image_Y.ravel(), residual_image_CbCr.ravel()])
pmf = stats_marg(merged_residuals, np.arange(-255,256))
huffman_coder = HuffmanCoder(lower_bound=-255)
huffman_coder.train(pmf)
#compressed, bitrate = huffman_coder.encode(merged_residuals)
#print(f"Compressed: {compressed}")
#print(f"Bitrate: {bitrate}")
#decoded_message = huffman_coder.decode(compressed, message_length=len(merged_residuals))
##print(np.array_equal(decoded_message, merged_residuals))

num_codewords, max_codeword_length, min_codeword_length = huffman_coder.get_codeword_stats()
print(f"Number of codewords: {num_codewords}")
print(f"Max. codeword length: {max_codeword_length}")
print(f"Min. codeword length: {min_codeword_length}")
