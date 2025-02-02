import numpy as np
from einops import rearrange

class ZeroRunCoder:

    def __init__(self, end_of_block=4000, block_size = 64):
        self.EOB = end_of_block
        self.block_size = block_size

    def encode(self, flat_patch_img: np.array):
        """
        This function gets a flattened patched image and produces a list of 
        symbols that applies a zero run encoding of the input where sequential
        blocks of zeroes (e.g. [... 0 0 0 0 0 ...]) are replaced with a marker zero
        and the number of additional zeroes (e.g. [... 0 4 ...]). The original sequence
        is processed in blocks of block_size and every encoding of a block ends with an
        end of block symbol. If all the original values are zero until the end of block,
        then no marker is necessary and we can put an EOB symbol directly.

        flat_patch_img: np.array of shape [H_patch, W_patch, C, Block_size]

        returns:
            encoded: List of symbols that represent the original elements
        
        """

        # YOUR CODE STARTS HERE
        flat_img = rearrange(flat_patch_img, 'h w c p -> (h w c) p', p=self.block_size)
        encoded = []

        # 遍历每个块
        for block in flat_img:
            block_encoded = []
            i = 0
            while i < self.block_size:
                if block[i] == 0:
                    # 计数连续的零
                    start = i
                    while i < self.block_size and block[i] == 0:
                        i += 1
                    zero_count = i - start
                    # 如果这一段零正好延伸到块尾，则直接用 EOB 表示全零剩余部分
                    if i == self.block_size:
                        block_encoded.append(self.EOB)
                        break
                    else:
                        # 否则用 [0, zero_count-1] 表示这一段零运行
                        block_encoded.extend([0, zero_count - 1])
                else:
                    # 非零值直接添加
                    block_encoded.append(block[i])
                    i += 1
            # 确保本块编码以 EOB 结尾（如果没有则补充）
            if len(block_encoded) == 0 or block_encoded[-1] != self.EOB:
                block_encoded.append(self.EOB)
            encoded.extend(block_encoded)


        # YOUR CODE ENDS HERE
        return encoded
    
    def decode(self, encoded, patch_shape):
        """
        This function gets an encoding and the original shape to decode the elements 
        of the original array. It acts as the inverse function of the decoder.

        encoded: List of symbols that represent the original elements
        patch_shape: List of 3 numbers that represent number of H_patch, W_patch and C

        returns:
            flat_patch_img: np.array of shape [H_patch, W_patch, C, Block_size]
        
        """
        
        # YOUR CODE STARTS HERE
        num_blocks = np.prod(patch_shape)
        # 初始化全 0 数组
        flat_img = np.zeros((num_blocks, self.block_size))
        pos = 0  # 编码序列的读取位置

        # 逐块解码，每块独立处理，直到遇到 EOB 标记
        for block_idx in range(num_blocks):
            curr_pos = 0  # 当前块内写入位置
            # 持续读取直到遇到 EOB
            while pos < len(encoded) and encoded[pos] != self.EOB:
                value = encoded[pos]
                if value == 0:
                    # 碰到 0 表示后面紧跟的数字表示零运行的长度（实际零数量 = marker 后数字 + 1）
                    if pos + 1 >= len(encoded):
                        raise ValueError("无效编码：零标记后缺少运行长度信息")
                    run_length = encoded[pos + 1] + 1
                    curr_pos += run_length  # 这段位置保持为 0
                    pos += 2
                else:
                    # 非零值直接写入当前位置
                    if curr_pos < self.block_size:
                        flat_img[block_idx, curr_pos] = value
                        curr_pos += 1
                    pos += 1
            # 跳过本块的 EOB 标记（如果存在）
            if pos < len(encoded) and encoded[pos] == self.EOB:
                pos += 1
            # 如果当前块内没有填满 block_size，其余部分默认为 0（flat_img 初始化为 0）
            if curr_pos > self.block_size:
                raise ValueError("decode error")
        # YOUR CODE ENDS HERE

        flat_patch_img = rearrange(
            flat_img, 
            '(h w c) p -> h w c p', 
            h = patch_shape[0], w =patch_shape[1],
            c =patch_shape[2], p=self.block_size)
        return flat_patch_img

        
        