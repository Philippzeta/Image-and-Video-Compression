from ivclab.utils import imread
from ivclab.entropy import stats_marg, calc_entropy
from ivclab.image import three_pixels_predictor
import numpy as np

# For this exercise, you need to implement three_pixels_predictor and
# _predict_from_neighbors functions in ivclab.image.predictive file.
# You can run ch2 tests to make sure they are implemented correctly

lena_img = imread(f'../../data/lena.tif')
residual_image_Y, residual_image_CbCr = three_pixels_predictor(lena_img, subsample_color_channels=False)
merged_residuals = np.concatenate([residual_image_Y.ravel(), residual_image_CbCr.ravel()])
pmf = stats_marg(merged_residuals, np.arange(-255,255))
entropy = calc_entropy(pmf)

print(f"Three pixels predictive coding entropy of lena.tif: H={entropy:.2f} bits/pixel")
