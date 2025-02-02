import numpy as np


class MotionCompensator:

    def __init__(self, search_range=4):
        self.search_range = search_range
        self.max_offset = 2 * search_range + 1

    def compute_motion_vector(self, ref_image, image):
        """
        Computes the motion vector that describes the motion between the single channeled reference image and the current image.
        The motion vector is represented as a 2D numpy array with shape [H / 8, W / 8, 1], where the x and y displacements
        are converted to a single value using the formula: motion_vector = y_displacement * (2 * search_range + 1) + x_displacement. Notice that
        displacements can take any value in the range [-search_range, search_range]. We compute the motion vectors only for
        the 8x8 non-overlapping blocks. Compute the closest indice using sum of the squared differences (SSD) between the reference block and the current block.

        ref_image: np.array of shape [H, W]
        image: np.array of shape [H, W]

        returns:
            motion_vector: np.array of shape [H / 8, W / 8, 1]
        """
        # YOUR CODE STARTS HERE
        H, W = image.shape
        block_size = 8
        H_blocks = H // block_size
        W_blocks = W // block_size
        motion_vector = np.zeros((H_blocks, W_blocks, 1), dtype=np.int32)

        for i in range(H_blocks):
            for j in range(W_blocks):
                # get current block position
                block_y = i * block_size
                block_x = j * block_size
                current_block = image[block_y:block_y + block_size, block_x:block_x + block_size]

                min_ssd = np.inf
                best_y = 0
                best_x = 0

                # 搜索最佳匹配块
                for dy in range(-self.search_range, self.search_range + 1):
                    for dx in range(-self.search_range, self.search_range + 1):
                        # calculate position in reference image
                        ref_y = block_y + dy
                        ref_x = block_x + dx

                        # 检查边界条件
                        if (ref_y >= 0 and ref_y + block_size <= H and
                                ref_x >= 0 and ref_x + block_size <= W):
                            ref_block = ref_image[ref_y:ref_y + block_size, ref_x:ref_x + block_size]
                            ssd = np.sum((current_block - ref_block) ** 2)

                            if ssd < min_ssd:
                                min_ssd = ssd
                                best_y = dy
                                best_x = dx

                # 将位移转换为非负索引（移出循环）
                dy_offset = best_y + self.search_range  # 将 [-search_range, search_range] 映射到 [0, 2*search_range]
                dx_offset = best_x + self.search_range

                # 计算一维索引
                mv = dy_offset * self.max_offset + dx_offset
                motion_vector[i, j, 0] = mv

                # YOUR CODE END HERE
        return motion_vector

    def reconstruct_with_motion_vector(self, ref_image, motion_vector):
        """
        Reconstructs the current image using the reference image and the motion vector. The motion vector is used to
        displace the 8x8 blocks in the reference image to their corresponding positions in the current image.

        ref_image: np.array of shape [H, W]
        motion_vector: np.array of shape [H / 8, W / 8, 1]

        returns:
            image: np.array of shape [H, W]
        """

        # image = np.zeros_like(ref_image)

        # YOUR CODE STARTS HERE
        H, W = ref_image.shape
        block_size = 8
        H_blocks, W_blocks = motion_vector.shape[:2]
        image = np.zeros_like(ref_image)

        for i in range(H_blocks):
            for j in range(W_blocks):
                y = i * block_size
                x = j * block_size
                mv = int(motion_vector[i, j, 0])

                # 正确解码位移量
                dy_offset = mv // self.max_offset
                dx_offset = mv % self.max_offset
                dy = dy_offset - self.search_range
                dx = dx_offset - self.search_range

                # 计算参考位置并处理边界
                ref_y = max(0, min(y + dy, H - block_size))
                ref_x = max(0, min(x + dx, W - block_size))

                # 复制参考块（自动处理边界）
                image[y:y + block_size, x:x + block_size] = \
                    ref_image[ref_y:ref_y + block_size, ref_x:ref_x + block_size]
        # YOUR CODE ENDS HERE
        return image