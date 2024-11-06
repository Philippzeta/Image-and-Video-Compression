import numpy as np
from ivclab.utils import imread
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

image = imread('../../data/satpic1.bmp')
image_np = np.array(image)

channels = [image_np[:, :, i] for i in range(3)]
kernel = np.array(
    [[1, 2, 1],
     [2, 4, 2],
     [1, 2, 1]]
)
kernel = kernel / kernel.sum()

# all chan
filtered_channels = [convolve2d(channel, kernel, mode='same', boundary='symm') for channel in channels]

# absolute diff
difference_channels = [np.abs(channel - filtered_channel) for channel, filtered_channel in zip(channels, filtered_channels)]

# 3 to 1
filtered_image = np.stack(filtered_channels, axis=-1).astype(np.uint8)
difference_image = np.stack(difference_channels, axis=-1)



# show diff
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image_np)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(filtered_image)
plt.title('Filtered Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(difference_image)
plt.title('Enhanced Difference Image')
plt.axis('off')

plt.tight_layout()
plt.show()
