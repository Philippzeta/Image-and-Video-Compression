import numpy as np
from ivclab.entropy import ZeroRunCoder
from ivclab.image import IntraCodec
from ivclab.entropy import HuffmanCoder, stats_marg
from ivclab.signal import rgb2ycbcr, ycbcr2rgb
from ivclab.video import MotionCompensator


class VideoCodec:

    def __init__(self,
                 quantization_scale=1.0,
                 bounds=(-1000, 4000),
                 end_of_block=4000,
                 block_shape=(8, 8),
                 search_range=4
                 ):
        self.quantization_scale = quantization_scale
        self.bounds = bounds
        self.end_of_block = end_of_block
        self.block_shape = block_shape
        self.search_range = search_range

        self.intra_codec = IntraCodec(quantization_scale=quantization_scale, bounds=bounds, end_of_block=end_of_block,
                                      block_shape=block_shape)
        self.motion_comp = MotionCompensator(search_range=search_range)
        self.motion_huffman = HuffmanCoder(lower_bound=((2 * search_range + 1) ** 2 - 1) // 2)

        self.encoding_first = True
        self.decoding_first = True
        self.prev_frame = None
        self.current_frame = None
        self.frame_shape=None
        self.mv_length=None
    def encode(self, frame):
        if self.encoding_first:
            self.encoding_first = False
            self.frame_shape = frame.shape
            # YOUR CODE STARTS HERE
            # 1. first use IntraCodec
            bitstream = self.intra_codec.intra_encode(frame)
            # 2. Decode and store as YCbCr reference frame
            decoded_frame = self.intra_codec.intra_decode(bitstream, frame.shape)
            self.prev_frame = rgb2ycbcr(decoded_frame)
            # YOUR CODE ENDS HERE
            return bitstream
        else:
            # YOUR CODE STARTS HERE
            # 1. RGB TO YCBCR
            current_ycbcr = rgb2ycbcr(frame)

            # 2. mv estimate

            mv = self.motion_comp.compute_motion_vector(self.prev_frame[:, :, 0], current_ycbcr[:, :, 0])

            # 3. predicate mv
            pred_y = self.motion_comp.reconstruct_with_motion_vector(self.prev_frame[:, :, 0], mv)
            pred_cb = self.motion_comp.reconstruct_with_motion_vector(self.prev_frame[:, :, 1], mv)
            pred_cr = self.motion_comp.reconstruct_with_motion_vector(self.prev_frame[:, :, 2], mv)
            pred_frame = np.stack([pred_y, pred_cb, pred_cr], axis=-1)

            # 4. redu
            err_im = current_ycbcr - pred_frame

            # 5. MV and redu Huffman
            #RED
            err_symbols = self.intra_codec.image2symbols(err_im, is_ycbcr=True)

            # 6. huffman encode
            if self.intra_codec.huffman.probs is None:
                err_range = np.arange(self.bounds[0], self.bounds[1] + 1)
                err_stats = stats_marg(np.array(err_symbols), err_range)
                self.intra_codec.huffman.train(err_stats)

            self.intra_codec.symbols_no = len(err_symbols)
            bytestream1, _ = self.intra_codec.huffman.encode(err_symbols)

            # MVencode
            if self.motion_huffman.probs is None:

                offset_mv = mv.flatten() + self.motion_huffman.lower_bound
                mv_range = np.arange(40, 121)
                mv_stats = np.zeros(len(mv_range))
                for i, val in enumerate(mv_range):
                    mv_stats[i] = np.sum(offset_mv == val)
                mv_stats += 1
                mv_stats = mv_stats / np.sum(mv_stats)
                self.motion_huffman.train(mv_stats)

            # Saves the length of the motion vector for decoding.
            self.mv_length = len(mv.flatten())
            offset_mv = mv.flatten() + self.motion_huffman.lower_bound
            bytestream2, _ = self.motion_huffman.encode(offset_mv)

            # 7. Updating the reference frame
            decoded_symbols = self.intra_codec.huffman.decode(bytestream1, self.intra_codec.symbols_no)
            decoded_err = self.intra_codec.symbols2image(decoded_symbols, current_ycbcr.shape, is_ycbcr=True)
            self.prev_frame = pred_frame + decoded_err
            # 8. Returns two bitstreams
            bitstream = (bytestream1, bytestream2)
            # YOUR CODE ENDS HERE
            return bitstream

    def decode(self, bitstream):
        if self.decoding_first:
            self.decoding_first = False
            frame = self.intra_codec.intra_decode(bitstream, self.frame_shape)
            self.current_frame = rgb2ycbcr(frame)
            return frame
        else:
            bytestream1, bytestream2 = bitstream

            # Decoding motion vectors
            mv_symbols = self.motion_huffman.decode(bytestream2, self.mv_length)
            mv_symbols = [x - self.motion_huffman.lower_bound for x in mv_symbols]

            H_blocks = self.current_frame.shape[0] // self.block_shape[0]
            W_blocks = self.current_frame.shape[1] // self.block_shape[1]
            mv = np.array(mv_symbols).reshape((H_blocks, W_blocks, 1))

            # reconstruction
            pred_y = self.motion_comp.reconstruct_with_motion_vector(self.current_frame[:, :, 0], mv)
            pred_cb = self.motion_comp.reconstruct_with_motion_vector(self.current_frame[:, :, 1], mv)
            pred_cr = self.motion_comp.reconstruct_with_motion_vector(self.current_frame[:, :, 2], mv)
            pred_frame = np.stack([pred_y, pred_cb, pred_cr], axis=-1)

            # decode redu
            decoded_symbols = self.intra_codec.huffman.decode(bytestream1, self.intra_codec.symbols_no)
            decoded_err = self.intra_codec.symbols2image(decoded_symbols, self.current_frame.shape, is_ycbcr=True)

            # rebuild frame
            reconstructed_frame = pred_frame + decoded_err

            # upload new frame
            self.current_frame = reconstructed_frame.copy()

            frame = ycbcr2rgb(reconstructed_frame)
            frame = np.clip(frame, 0, 255)
            # YOUR CODE ENDS HERE
            return frame