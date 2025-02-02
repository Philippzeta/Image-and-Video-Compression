# Implement the MotionCompensator and VideoCodec classes from ivclab.video.motion and ivclab.video.videocodec respectively.
# Run the Video Codec on the provided video sequence and compute the average PSNR of the reconstructed frames with respect to the original frames.
# Plot your results against the provided "ground truth" values.
import numpy as np
import matplotlib.pyplot as plt
from ivclab.image import IntraCodec
from ivclab.video import VideoCodec
from ivclab.utils import imread, calc_psnr

def process_video_sequence(q_scale):
    # 1. READ AND TRAIN
    lena_small = np.double(imread('../../data/lena_small.tif'))

    # 2.  VideoCodec
    video_codec = VideoCodec(quantization_scale=q_scale,
                           bounds=(-1000, 4000),
                           end_of_block=4000,
                           block_shape=(8, 8),
                           search_range=4)
    video_codec.intra_codec.train_huffman_from_image(lena_small)

    # 3. ALL VIDEO
    filenames = [
        '../../videodata/foreman0020.bmp',  # FIRST USE IF THEN USE ELSE
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

        # BITRATE
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

    # 4. AVERAGE
    avg_psnr = np.mean(psnr_list)
    avg_bitrate = np.mean(bitrate_list)

    return avg_psnr, avg_bitrate

def plot_combined_results(video_bitrates, video_psnrs, still_bitrates, still_psnrs):
    plt.figure(figsize=(12, 8))

    # PLOT
    plt.plot(video_bitrates, video_psnrs, '-', color='blue', linewidth=2, label='Video Codec')
    plt.scatter(video_bitrates, video_psnrs, color='blue', s=100, zorder=5)

    # MARK
    for i, (x, y) in enumerate(zip(video_bitrates, video_psnrs)):
        plt.annotate(f'({x:.2f}, {y:.2f})',
                     (x, y),
                     xytext=(10, 10),
                     textcoords='offset points',
                     fontsize=8,
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    # RESULT
    plt.plot(still_bitrates, still_psnrs, '-', color='red', linewidth=2, label='Still Image Codec')
    plt.scatter(still_bitrates, still_psnrs, color='red', s=100, zorder=5)

    # MARK
    for i, (x, y) in enumerate(zip(still_bitrates, still_psnrs)):
        plt.annotate(f'({x:.2f}, {y:.2f})',
                     (x, y),
                     xytext=(-60, -15),
                     textcoords='offset points',
                     fontsize=8,
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    plt.xlabel('Bitrate [bit/pixel]')
    plt.ylabel('PSNR [dB]')
    plt.title('Operational D-R Curve Comparison')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')

    plt.savefig('dr_curve_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # TEST
    video_q_scales = [0.07, 0.2, 0.4, 0.8, 1.0, 1.5, 2, 3, 4, 4.5]
    video_psnr_values = []
    video_bitrate_values = []

    for q_scale in video_q_scales:
        print(f"\nProcessing Video Codec with q_scale = {q_scale}")
        avg_psnr, avg_bitrate = process_video_sequence(q_scale)
        video_psnr_values.append(avg_psnr)
        video_bitrate_values.append(avg_bitrate)

    # TEST
    still_q_scales = [0.15, 0.3, 0.7, 1.0, 1.5, 3, 5, 7, 10]
    still_psnr_values = []
    still_bitrate_values = []

    # READ IMAGE
    lena_small = np.double(imread('../../data/lena_small.tif'))
    lena = np.double(imread('../../data/lena.tif'))

    for scale in still_q_scales:
        print(f"\nProcessing Still Image Codec with scale = {scale}")
        codec = IntraCodec(
            quantization_scale=scale,
            bounds=(-1000, 4000),
            end_of_block=4000,
            block_shape=(8, 8)
        )
        codec.train_huffman_from_image(lena_small)

        bitstream = codec.intra_encode(lena)
        reconstructed_img = codec.intra_decode(bitstream, lena.shape)

        psnr = calc_psnr(lena, reconstructed_img)
        bitrate = len(bitstream) * 8 / (lena.shape[0] * lena.shape[1] / 3)

        still_psnr_values.append(psnr)
        still_bitrate_values.append(bitrate)

    video_sorted = np.argsort(video_bitrate_values)
    still_sorted = np.argsort(still_bitrate_values)

    # PLOT PICTURE
    plot_combined_results(
        np.array(video_bitrate_values)[video_sorted],
        np.array(video_psnr_values)[video_sorted],
        np.array(still_bitrate_values)[still_sorted],
        np.array(still_psnr_values)[still_sorted]
    )


if __name__ == '__main__':
    main()