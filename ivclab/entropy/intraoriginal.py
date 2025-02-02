import numpy as np
from einops import rearrange


class ZeroRunCoder:

    def __init__(self, end_of_block=4000, block_size=64):
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

        for block in flat_img:
            block_encoded = []
            zero_count = 0
            current_pos = 0  # 跟踪当前处理位置

            for value in block:
                if current_pos >= self.block_size:
                    break

                if value == 0:
                    zero_count += 1
                else:
                    # 处理之前的零
                    if zero_count > 0:
                        available = self.block_size - current_pos
                        run_length = min(zero_count, available)

                        if run_length > 0:
                            block_encoded.extend([0, run_length - 1])
                            current_pos += run_length
                            zero_count -= run_length

                    # 处理当前非零值
                    if current_pos < self.block_size:
                        block_encoded.append(value)
                        current_pos += 1
                    zero_count = 0  # 重置零计数

            # 处理末尾剩余零
            if zero_count > 0:
                available = self.block_size - current_pos
                run_length = min(zero_count, available)

                if run_length == self.block_size:  # 全零块
                    block_encoded = [self.EOB]
                elif run_length > 0:
                    block_encoded.extend([0, run_length - 1])
                    current_pos += run_length

            # 添加EOB并检查块完整性
            if len(block_encoded) == 0 or block_encoded[-1] != self.EOB:
                # 检查是否已填满整个块
                if current_pos < self.block_size:
                    remaining = self.block_size - current_pos
                    if remaining > 0:
                        block_encoded.extend([0, remaining - 1])
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
        flat_img = np.zeros((num_blocks, self.block_size))
        block_idx = 0
        pos = 0

        while block_idx < num_blocks and pos < len(encoded):
            if encoded[pos] == self.EOB:
                block_idx += 1
                pos += 1
                continue

            curr_pos = 0
            while curr_pos < self.block_size and pos < len(encoded):
                value = encoded[pos]

                if value == self.EOB:
                    block_idx += 1
                    pos += 1
                    break

                if value == 0:
                    count = encoded[pos + 1] + 1  # 恢复实际零数量
                    end = min(curr_pos + count, self.block_size)
                    curr_pos = end
                    pos += 2
                else:
                    if curr_pos < self.block_size:
                        flat_img[block_idx, curr_pos] = value
                        curr_pos += 1
                    pos += 1

            # 处理可能缺失的EOB
            while curr_pos < self.block_size:
                flat_img[block_idx, curr_pos] = 0
                curr_pos += 1
        # YOUR CODE ENDS HERE

        flat_patch_img = rearrange(
            flat_img,
            '(h w c) p -> h w c p',
            h=patch_shape[0], w=patch_shape[1],
            c=patch_shape[2], p=self.block_size)
        return flat_patch_img

