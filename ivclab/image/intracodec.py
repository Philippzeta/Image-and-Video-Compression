import numpy as np
from ivclab.entropy import ZeroRunCoder
from ivclab.quantization import PatchQuant
from ivclab.utils import ZigZag,Patcher
from ivclab.signal import DiscreteCosineTransform
from ivclab.entropy import HuffmanCoder, stats_marg
from ivclab.signal import rgb2ycbcr, ycbcr2rgb

class IntraCodec:

    def __init__(self, 
                 quantization_scale = 1.0,
                 bounds = (-1000, 4000),
                 end_of_block = 4000,
                 block_shape = (8,8)
                 ):
        self.dct = DiscreteCosineTransform()
        self.quant = PatchQuant(quantization_scale=quantization_scale)
        self.zigzag = ZigZag()
        self.zerorun = ZeroRunCoder(end_of_block=end_of_block, block_size= block_shape[0] * block_shape[1])
        self.huffman = HuffmanCoder(lower_bound=bounds[0])

    def image2symbols(self, img: np.array):
        """
        Computes the symbol representation of an image by applying rgb2ycbcr,
        DCT, Quantization, ZigZag and ZeroRunEncoding in order.

        img: np.array of shape [H, W, C]

        returns:
            symbols: List of integers
        """
        # YOUR CODE STARTS HERE
        # 1. RGB转YCbCr
        ycbcr = rgb2ycbcr(img)
        patcher = Patcher()
        patched_img = patcher.patch(ycbcr)
        # 2. 应用DCT变换
        dct_coeffs = self.dct.transform(patched_img)

        # 3. 量化
        quantized = self.quant.quantize(dct_coeffs)

        # 4. ZigZag扫描
        zigzagged = self.zigzag.flatten(quantized)

        # 5. 零游程编码
        symbols = self.zerorun.encode(zigzagged)
        # YOUR CODE ENDS HERE
        return symbols
    
    def symbols2image(self, symbols, original_shape):
        """
        Reconstructs the original image from the symbol representation
        by applying ZeroRunDecoding, Inverse ZigZag, Dequantization and 
        IDCT, ycbcr2rgb in order. The argument original_shape is required to compute 
        patch_shape, which is needed by ZeroRunDecoding to correctly 
        reshape the input image from blocks.

        symbols: List of integers
        original_shape: List of 3 elements that contains H, W and C
        
        returns:
            reconstructed_img: np.array of shape [H, W, C]
        """
        patch_shape = [original_shape[0] // 8, original_shape[1] // 8, original_shape[2]]
        # YOUR CODE STARTS HERE
        # 1. 零游程解码
        zigzagged = self.zerorun.decode(symbols, patch_shape)

        # 2. 反ZigZag扫描
        quantized = self.zigzag.unflatten(zigzagged)

        # 3. 反量化
        dct_coeffs = self.quant.dequantize(quantized)

        # 4. 反DCT变换
        ycbcr_patched = self.dct.inverse_transform(dct_coeffs)

        # 5. YCbCr转RGB
        patcher = Patcher()
        ycbcr = patcher.unpatch(ycbcr_patched)
        reconstructed_img = ycbcr2rgb(ycbcr)
        # YOUR CODE ENDS HERE
        return reconstructed_img
    
    def train_huffman_from_image(self, training_img):
        """
        Finds the symbols representing the image, extracts the 
        probability distribution of them and trains the huffman coder with it.

        training_img: np.array of shape [H, W, C]

        returns:
            Nothing
        """
        # YOUR CODE STARTS HERE
        # 1. 获取图像的符号表示
        symbols = self.image2symbols(training_img)

        # 2. 计算符号的概率分布
        symbols_array = np.array(symbols)
        prob_dist = stats_marg(symbols_array, pixel_range=5000)

        # 3. 训练Huffman编码器
        self.huffman.train(prob_dist)
        # YOUR CODE ENDS HERE

    def intra_encode(self, img: np.array):
        """
        Encodes an image to a bitstream and return it by converting it to
        symbols and compressing them with the Huffman coder.

        img: np.array of shape [H, W, C]

        returns:
            bitstream: List of integers produced by the Huffman coder
        """
        # YOUR CODE STARTS HERE
        # 1. 将图像转换为符号
        symbols = self.image2symbols(img)

        # 2. 使用Huffman编码器进行编码
        compressed, _ = self.huffman.encode(symbols)
        # 3. 确保返回的是列表形式
        bitstream = compressed.astype(np.uint8).tolist()  # 转换为uint8类型再转列表
        # YOUR CODE ENDS HERE
        return bitstream

    def intra_decode(self, bitstream, original_shape):
        """
        Decodes an image from a bitstream by decoding it with the Huffman
        coder and reconstructing it from the symbols.

        bitstream: List of integers produced by the Huffman coder
        original_shape: List of 3 values that contain H, W, and C

        returns:
            reconstructed_img: np.array of shape [H, W, C]
        """
        # YOUR CODE STARTS HERE
        # 1. 计算期望的符号数量
        expected_symbols = (original_shape[0] * original_shape[1] * original_shape[2] * 64) // (8 * 8)

        # 2. 使用Huffman解码器解码
        # 确保输入格式正确
        compressed = np.array(bitstream, dtype=np.uint8)  # 使用uint8类型
        symbols = self.huffman.decode(compressed, message_length=expected_symbols)

        # 3. 从符号重建图像
        reconstructed_img = self.symbols2image(symbols, original_shape)
        # YOUR CODE ENDS HERE
        return reconstructed_img

