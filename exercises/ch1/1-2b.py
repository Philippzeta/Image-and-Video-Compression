import numpy as np
import matplotlib.pyplot as plt

# use kernel from 1-2 a
kernel = np.asarray(
    [[1, 2, 1],
     [2, 4, 2],
     [1, 2, 1]]
)

# normalization
kernel = kernel / kernel.sum()


fft_result = np.fft.fft2(kernel, s=(1000, 1000))
fft_shifted = np.fft.fftshift(fft_result)

magnitude_spectrum = np.abs(fft_shifted)


plt.imshow(magnitude_spectrum, cmap='jet')
plt.title('Frequency Response of the Filter')
plt.colorbar()
plt.show()
