from ivclab.utils import imread
import numpy as np
import matplotlib.pyplot as plt

image = imread('../../data/lena_small.tif')

red_channel = image[:, :, 0]
green_channel = image[:, :, 1]
blue_channel = image[:, :, 2]

counters_red = np.zeros(256, dtype=int)
counters_green = np.zeros(256, dtype=int)
counters_blue = np.zeros(256, dtype=int)


counters_red = np.bincount(red_channel.flatten(), minlength=256)
counters_green = np.bincount(green_channel.flatten(), minlength=256)
counters_blue = np.bincount(blue_channel.flatten(), minlength=256)


plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.bar(range(256), counters_red, color='red')
plt.title('Red Channel Histogram')

plt.subplot(3, 1, 2)
plt.bar(range(256), counters_green, color='green')
plt.title('Green Channel Histogram')

plt.subplot(3, 1, 3)
plt.bar(range(256), counters_blue, color='blue')
plt.title('Blue Channel Histogram')

plt.tight_layout()
plt.show()
