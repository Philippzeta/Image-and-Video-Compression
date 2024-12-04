import unittest
import numpy as np
from ivclab.utils import imread, calc_psnr
from ivclab.quantization.quantizers import lloyd_max, inv_lloyd_max, uniquant, inv_uniquant, vector_quantizer,inv_vector_quantizer, apply_vector_quantizer
from ivclab.entropy.huffman import HuffmanCoder


class TestMeasurements(unittest.TestCase):

    def setUp(self) -> None:
        self.orig_img = imread('../../data/lena.tif')
        self.ref_img = imread('../../data/satpic1.bmp')
        return super().setUp()

    def test_correct_psnr_uniquant(self) -> None:
        qImage_orig = uniquant(self.orig_img, 4)
        qImage_ref = uniquant(self.ref_img, 5)

        recImage_orig = inv_uniquant(qImage_orig, 4)
        recImage_ref = inv_uniquant(qImage_ref, 5)

        psnr_orig = calc_psnr(recImage_orig, self.orig_img)
        psnr_ref = calc_psnr(recImage_ref, self.ref_img)
        self.assertAlmostEqual(psnr_orig, 34.69, delta=0.2)
        self.assertAlmostEqual(psnr_ref, 40.72, delta=0.2)

    def test_correct_psnr_lloydmax(self) -> None:
        qImage_orig, clusters_orig = lloyd_max(self.orig_img, 4, 1e-5)
        qImage_ref, clusters_ref = lloyd_max(self.ref_img, 5, 1e-5)

        recImage_orig = inv_lloyd_max(qImage_orig, clusters_orig)
        recImage_ref = inv_lloyd_max(qImage_ref, clusters_ref)

        psnr_orig = calc_psnr(recImage_orig, self.orig_img)
        psnr_ref = calc_psnr(recImage_ref, self.ref_img)
        self.assertAlmostEqual(psnr_orig, 35.44, delta=0.2)
        self.assertAlmostEqual(psnr_ref, 41.05, delta=0.2)

    def test_correct_cluster_lloydmax(self) -> None:
        pass

    @staticmethod
    def vq_pipeline():
        img_small = imread('../../data/lena_small.tif')
        clusters, _ = vector_quantizer(img_small, 3, 0.1, 2)
        qImage_small = apply_vector_quantizer(img_small, clusters, 2)

        pmf = np.histogram(qImage_small.flatten(), bins=np.arange(0, 2 ** 3 + 1), density=True)[0]

        huffman_coder = HuffmanCoder(lower_bound=0)
        huffman_coder.train(probs=pmf)

        img_lena = imread('../../data/lena.tif')
        qImage = apply_vector_quantizer(img_lena, clusters, 2)

        bytestream, bitrate = huffman_coder.encode(qImage.flatten())
        k_rec = huffman_coder.decode(bytestream, qImage.size)

        qReconst_image = k_rec.reshape(qImage.shape)
        return inv_vector_quantizer(qReconst_image, clusters, 2)

    def test_correct_psnr_vq(self) -> None:
        recImage_orig = self.vq_pipeline()
        # recImage_ref = self.vq_pipeline(self.ref_img)

        psnr_orig = calc_psnr(recImage_orig, self.orig_img)
        # psnr_ref = calc_psnr(recImage_ref, self.ref_img)
        self.assertAlmostEqual(psnr_orig, 26.62, delta=0.2)
        # self.assertAlmostEqual(psnr_ref, 41.05, delta=0.2)


if __name__ == '__main__':
    unittest.main()