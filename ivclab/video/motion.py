import numpy as np

class MotionCompensator:

    def __init__(self, search_range=4):
        self.search_range = search_range

    def compute_motion_vector(self,ref_image, image):
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
        motion_vector = np.zeros((H_blocks, W_blocks, 1))

        # 遍历每个8x8块
        for i in range(H_blocks):
            for j in range(W_blocks):
                # 当前块的坐标
                y = i * block_size
                x = j * block_size

                # 提取当前块
                current_block = image[y:y + block_size, x:x + block_size]

                min_ssd = float('inf')
                best_dx = 0
                best_dy = 0

                # 在搜索范围内寻找最佳匹配
                for dy in range(-self.search_range, self.search_range + 1):
                    for dx in range(-self.search_range, self.search_range + 1):
                        # 计算参考块的坐标
                        ref_y = y + dy
                        ref_x = x + dx

                        # 检查边界条件
                        if (ref_y >= 0 and ref_y + block_size <= H and
                                ref_x >= 0 and ref_x + block_size <= W):

                            # 提取参考块
                            ref_block = ref_image[ref_y:ref_y + block_size,
                                        ref_x:ref_x + block_size]

                            # 计算SSD
                            ssd = np.sum((current_block - ref_block) ** 2)

                            # 更新最小SSD和最佳位移
                            if ssd < min_ssd:
                                min_ssd = ssd
                                best_dy = dy
                                best_dx = dx

                # 将(dx,dy)转换为单一值
                # motion_vector = y_displacement * (2 * search_range + 1) + x_displacement
                mv = best_dy * (2 * self.search_range + 1) + best_dx
                motion_vector[i, j, 0] = mv
            # YOUR CODE stop HERE
        return motion_vector
    
    def reconstruct_with_motion_vector(self,ref_image, motion_vector):
        """
        Reconstructs the current image using the reference image and the motion vector. The motion vector is used to
        displace the 8x8 blocks in the reference image to their corresponding positions in the current image.

        ref_image: np.array of shape [H, W]
        motion_vector: np.array of shape [H / 8, W / 8, 1]

        returns:
            image: np.array of shape [H, W]
        """

        #image = np.zeros_like(ref_image)

        # YOUR CODE STARTS HERE
        H, W = ref_image.shape
        block_size = 8
        H_blocks = H // block_size
        W_blocks = W // block_size
        image = np.zeros_like(ref_image)

        # 遍历每个8x8块
        for i in range(H_blocks):
            for j in range(W_blocks):
                # 获取当前块的运动矢量值
                mv = motion_vector[i, j, 0]

                # 将单一值解码回(dx,dy)位移
                # mv = dy * (2 * search_range + 1) + dx
                dy = mv // (2 * self.search_range + 1)
                dx = mv % (2 * self.search_range + 1)

                # 如果dx或dy超过search_range，说明是负位移
                if dx > self.search_range:
                    dx -= (2 * self.search_range + 1)
                if dy > self.search_range:
                    dy -= (2 * self.search_range + 1)

                # 计算当前块和参考块的坐标
                y = i * block_size
                x = j * block_size
                ref_y = y + dy
                ref_x = x + dx

                # 检查边界条件
                if (ref_y >= 0 and ref_y + block_size <= H and
                        ref_x >= 0 and ref_x + block_size <= W):
                    # 从参考帧复制对应的块
                    image[y:y + block_size, x:x + block_size] = \
                        ref_image[ref_y:ref_y + block_size, ref_x:ref_x + block_size]
        # YOUR CODE ENDS HERE
        return image