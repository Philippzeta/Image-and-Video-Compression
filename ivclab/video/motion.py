import numpy as np

class MotionCompensator:

    def __init__(self, search_range=4):
        self.search_range = search_range
        self.max_offset = 2 * search_range + 1

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
        motion_vector = np.zeros((H_blocks, W_blocks, 1), dtype=np.int32)

        # Create index table for motion vector encoding
        index_table = np.arange(1, 82).reshape(9, 9).T

        # Block positions
        block_positions_x = range(0, W, block_size)
        block_positions_y = range(0, H, block_size)

        for i in range(H_blocks):
            for j in range(W_blocks):
                # Current block position
                loc_x = block_positions_x[j]
                loc_y = block_positions_y[i]
                current_block = image[loc_y:loc_y + block_size, loc_x:loc_x + block_size]

                min_ssd = None  #
                best_x = None  # 初始化为None
                best_y = None

                # Search in ±4 range
                for ref_x in range(max(0, loc_x - self.search_range),
                                   min(W - block_size + 1, loc_x + self.search_range + 1)):
                    for ref_y in range(max(0, loc_y - self.search_range),
                                       min(H - block_size + 1, loc_y + self.search_range + 1)):
                        # Get reference block
                        ref_block = ref_image[ref_y:ref_y + block_size, ref_x:ref_x + block_size]

                        # Calculate SSD
                        ssd = np.sum((current_block - ref_block) ** 2)

                        # 第一个有效块或找到更好的匹配
                        if min_ssd is None or ssd <= min_ssd:
                            min_ssd = ssd
                            best_x = ref_x
                            best_y = ref_y

                # Convert to relative vector and get index from table
                vector_x = best_x - loc_x + self.search_range
                vector_y = best_y - loc_y + self.search_range

                # Get motion vector index from the table
                mv = index_table[vector_y, vector_x]

                motion_vector[i, j, 0] = mv - 1  # Subtract 1 for 0-based indexing
                # YOUR CODE END HERE
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

        # YOUR CODE STARTS HERE
        H, W = ref_image.shape
        block_size = 8
        H_blocks, W_blocks = motion_vector.shape[:2]
        image = np.zeros_like(ref_image)

        # Create index table
        index_table = np.arange(1, 82).reshape(9, 9).T

        for i in range(H_blocks):
            for j in range(W_blocks):
                # Current block position
                loc_x = j * block_size
                loc_y = i * block_size

                # Get motion vector and convert to relative position
                mv = int(motion_vector[i, j, 0]) + 1  # Add 1 for 1-based indexing
                vector_y, vector_x = np.where(index_table == mv)
                vector_y = vector_y[0] - self.search_range  # Center to origin
                vector_x = vector_x[0] - self.search_range

                # Calculate reference position
                ref_x = loc_x + vector_x
                ref_y = loc_y + vector_y

                # Ensure within bounds
                ref_x = max(0, min(ref_x, W - block_size))
                ref_y = max(0, min(ref_y, H - block_size))

                # Copy block
                image[loc_y:loc_y + block_size, loc_x:loc_x + block_size] = \
                    ref_image[ref_y:ref_y + block_size, ref_x:ref_x + block_size]
        # YOUR CODE ENDS HERE
        return image