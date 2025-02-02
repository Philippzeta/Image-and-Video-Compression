import numpy as np
import matplotlib.pyplot as plt
from ivclab.image import IntraCodec
from ivclab.video import VideoCodec
from ivclab.utils import imread, calc_psnr


def process_video_sequence(q_scale):
    # 1. 读取训练图像与第一帧
    lena_small = np.double(imread('../../data/lena_small.tif'))

    # 2. 初始化 VideoCodec
    video_codec = VideoCodec(quantization_scale=q_scale,
                             bounds=(-1000, 4000),
                             end_of_block=4000,
                             block_shape=(8, 8),
                             search_range=4)
    video_codec.intra_codec.train_huffman_from_image(lena_small)

    # 4. 处理后续帧
    filenames = [

        '../../videodata/foreman0021.bmp',
        '../../videodata/foreman0022.bmp',
        '../../videodata/foreman0023.bmp',
        '../../videodata/foreman0024.bmp',
        '../../videodata/foreman0025.bmp',
        '../../videodata/foreman0026.bmp',
        '../../videodata/foreman0027.bmp',
        '../../videodata/foreman0028.bmp',
        '../../videodata/foreman0029.bmp',
        '../../videodata/foreman0030.bmp',
        '../../videodata/foreman0031.bmp',
        '../../videodata/foreman0032.bmp',
        '../../videodata/foreman0033.bmp',
        '../../videodata/foreman0034.bmp',
        '../../videodata/foreman0035.bmp',
        '../../videodata/foreman0036.bmp',
        '../../videodata/foreman0037.bmp',
        '../../videodata/foreman0038.bmp',
        '../../videodata/foreman0039.bmp',
        '../../videodata/foreman0040.bmp'
    ]


    psnr_list = []
    bitrate_list = []

    for fname in filenames:
        current_frame = np.double(imread(fname))
        bitstream = video_codec.encode(current_frame)
        reconstructed_frame = video_codec.decode(bitstream)
        psnr_val = calc_psnr(current_frame, reconstructed_frame)

        # 计算比特率
        total_bits = 0
        if isinstance(bitstream, tuple):
            for part in bitstream:
                total_bits += len(part) * 8
        else:
            total_bits = len(bitstream) * 8
        total_pixels = current_frame.shape[0] * current_frame.shape[1]
        bitrate_val = total_bits / (total_pixels / 3)

        psnr_list.append(psnr_val)
        bitrate_list.append(bitrate_val)
        print(f"Frame {fname}: PSNR = {psnr_val:.2f} dB, Bitrate = {bitrate_val:.2f} bpp")

        # 4. 计算平均值
    avg_psnr = np.mean(psnr_list)
    avg_bitrate = np.mean(bitrate_list)

    return avg_psnr, avg_bitrate


def main():
    # 定义不同的量化参数
    q_scales = [0.07, 0.2, 0.4, 0.8, 1.0, 1.5, 2, 3, 4, 4.5]
    psnr_values = []
    bitrate_values = []

    # 对每个量化参数进行测试
    for q_scale in q_scales:
        print(f"\nProcessing with q_scale = {q_scale}")
        avg_psnr, avg_bitrate = process_video_sequence(q_scale)
        psnr_values.append(avg_psnr)
        bitrate_values.append(avg_bitrate)
        print(f"Average PSNR = {avg_psnr:.2f} dB, Average Bitrate = {avg_bitrate:.2f} bpp")

    # 按比特率排序
    sorted_indices = np.argsort(bitrate_values)
    sorted_bitrates = np.array(bitrate_values)[sorted_indices]
    sorted_psnrs = np.array(psnr_values)[sorted_indices]

    # 打印排序后的结果
    print("\nSorted Results:")
    for i in range(len(sorted_bitrates)):
        print(f"Bitrate = {sorted_bitrates[i]:.2f} bpp, PSNR = {sorted_psnrs[i]:.2f} dB")

    # 绘制 D-R 曲线
    plt.figure(figsize=(10, 8))

    # 绘制线条和点
    plt.plot(sorted_bitrates, sorted_psnrs, 'bo-')
    plt.xlabel('Bitrate [bit/pixel]')
    plt.ylabel('PSNR [dB]')
    plt.title('Operational D-R Curve')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()