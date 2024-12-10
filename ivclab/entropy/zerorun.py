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
        flat_img = rearrange(flat_patch_img, 'h w c p -> (h w c) p', p=self.block_size)
        # YOUR CODE STARTS HERE
        flat_img = rearrange(flat_patch_img, 'h w c p -> (h w c) p', p=self.block_size)
        encoded = []

        # 处理每个块
        for block in flat_img:
            block_encoded = []
            zero_count = 0

            for value in block:
                if value == 0:
                    zero_count += 1
                else:
                    # 如果之前有零,先编码零的数量
                    if zero_count > 0:
                        block_encoded.extend([0, zero_count])
                        zero_count = 0
                    block_encoded.append(value)

            # 处理块末尾的零
            if zero_count > 0:
                # 如果块末尾全是零,直接使用EOB
                if len(block_encoded) == 0:
                    block_encoded.append(self.EOB)
                else:
                    block_encoded.extend([0, zero_count])

            # 添加块结束符
            if block_encoded[-1] != self.EOB:
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
        original_shape = patch_shape
        num_blocks = patch_shape[0] * patch_shape[1] * patch_shape[2]
        flat_img = np.zeros((num_blocks, self.block_size))

        block_idx = 0  # 当前块的索引
        pos = 0  # 编码列表的当前位置
        curr_pos = 0  # 当前块内的位置

        while block_idx < num_blocks:
            if pos >= len(encoded):
                break

            value = encoded[pos]

            # 如果遇到EOB标记
            if value == self.EOB:
                block_idx += 1  # 移至下一块
                curr_pos = 0  # 重置块内位置
                pos += 1
                continue

            # 处理零游程编码
            if value == 0:
                zero_count = encoded[pos + 1]
                curr_pos += zero_count
                pos += 2
            else:
                flat_img[block_idx, curr_pos] = value
                curr_pos += 1
                pos += 1
        # YOUR CODE ENDS HERE

        flat_patch_img = rearrange(
            flat_img, 
            '(h w c) p -> h w c p', 
            h = original_shape[0], w = original_shape[1], 
            c = original_shape[2], p=self.block_size)
        return flat_patch_img

        
        