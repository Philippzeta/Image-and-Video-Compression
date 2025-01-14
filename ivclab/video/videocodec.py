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
        """
        编码视频帧。第一帧使用帧内编码，后续帧使用运动补偿和残差编码。
        """
        if self.encoding_first:
            # 处理第一帧
            self.encoding_first = False

            # 使用帧内编码器直接编码第一帧
            bitstream = self.intra_codec.intra_encode(frame)

            # 保存当前帧作为参考帧
            self.prev_frame = frame

            return bitstream

        else:
            # 处理后续帧
            # 1. 转换到YCbCr颜色空间
            current_ycbcr = rgb2ycbcr(frame)
            prev_ycbcr = rgb2ycbcr(self.prev_frame)

            # 2. 对每个通道分别进行运动估计
            motion_vectors = []
            residuals = []
            for c in range(3):  # Y, Cb, Cr通道
                # 计算运动矢量
                mv = self.motion_comp.compute_motion_vector(
                    prev_ycbcr[:, :, c],
                    current_ycbcr[:, :, c]
                )
                motion_vectors.append(mv)

                # 使用运动矢量重建帧
                predicted = self.motion_comp.reconstruct_with_motion_vector(
                    prev_ycbcr[:, :, c],
                    mv
                )

                # 计算残差
                residual = current_ycbcr[:, :, c] - predicted
                residuals.append(residual)

            # 3. 将运动矢量和残差合并为3D数组
            motion_vectors = np.stack(motion_vectors, axis=-1)
            residuals = np.stack(residuals, axis=-1)

            # 4. 对运动矢量进行零游程编码
            zrc = ZeroRunCoder(end_of_block=self.end_of_block,
                               block_size=self.block_shape[0] * self.block_shape[1])
            mv_symbols = zrc.encode(motion_vectors.flatten())

            # 5. 对运动矢量进行Huffman编码
            mv_prob = stats_marg(mv_symbols)
            self.motion_huffman.train(mv_prob)
            mv_bits = self.motion_huffman.encode(mv_symbols)[0]

            # 6. 使用帧内编码器编码残差
            residual_bits = self.intra_codec.intra_encode(residuals)

            # 7. 合并比特流
            mv_length = len(mv_bits)
            length_info = str(mv_length).encode() + b'|'  # 使用'|'作为分隔符
            bitstream = length_info + mv_bits + residual_bits

            # 8. 更新参考帧
            self.prev_frame = frame
            return bitstream

    def decode(self, bitstream):
        """
        解码视频帧。第一帧使用帧内解码，后续帧使用运动补偿和残差解码。
        """
        if self.decoding_first:
            # 处理第一帧
            self.decoding_first = False

            # 使用帧内解码器直接解码第一帧
            frame = self.intra_codec.intra_decode(bitstream, self.prev_frame.shape)

            # 保存解码后的帧作为参考帧
            self.prev_frame = frame

            return frame

        else:
            # 处理后续帧
            # 1. 分离运动矢量和残差的比特流
            length_info, data = bitstream.split(b'|', 1)  # 分离长度信息
            mv_length = int(length_info.decode())

            # 根据长度信息准确分离运动矢量和残差比特流
            mv_bits = data[:mv_length]
            residual_bits = data[mv_length:]

            # 2. 解码运动矢量
            H, W, C = self.prev_frame.shape
            mv_shape = (H // 8, W // 8, C)
            mv_size = mv_shape[0] * mv_shape[1] * mv_shape[2]

            # 2.1 Huffman解码
            mv_symbols = self.motion_huffman.decode(mv_bits, mv_size)

            # 2.2 零游程解码
            zrc = ZeroRunCoder(end_of_block=self.end_of_block,
                               block_size=self.block_shape[0] * self.block_shape[1])
            motion_vectors = zrc.decode(mv_symbols, mv_shape)

            # 3. 解码残差
            residuals = self.intra_codec.intra_decode(residual_bits, self.prev_frame.shape)

            # 4. 转换参考帧到YCbCr空间
            prev_ycbcr = rgb2ycbcr(self.prev_frame)

            # 5. 对每个通道进行运动补偿和残差重建
            reconstructed_ycbcr = np.zeros_like(prev_ycbcr)
            for c in range(3):
                # 使用运动矢量重建预测帧
                predicted = self.motion_comp.reconstruct_with_motion_vector(
                    prev_ycbcr[:, :, c],
                    motion_vectors[:, :, c:c + 1]
                )

                # 添加残差
                reconstructed_ycbcr[:, :, c] = predicted + residuals[:, :, c]

            # 6. 转换回RGB空间
            frame = ycbcr2rgb(reconstructed_ycbcr)

            # 7. 更新参考帧
            self.prev_frame = frame

            return frame
    

    

