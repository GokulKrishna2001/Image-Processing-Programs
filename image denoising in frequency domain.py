#image denoising using fourier transform in frequency domain
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import fftn, ifftn, fftshift, ifftshift
fig=plt.figure(figsize=(10,10))

image=cv2.imread('images/peppers_color_256.tif',0)

#adding gaussian noise(mean,sd,image)
noise=np.random.normal(1,20,image.shape)
noisy_image=image+noise
noisy_image = np.uint8(noisy_image)


fig.add_subplot(221)
plt.imshow(image,cmap="gray")
plt.title("Original image")

fig.add_subplot(222)
plt.imshow(noisy_image,cmap="gray")
plt.title('Noisy Image')

#FFT of the noisy image
FFT=fftn(noisy_image)
FFT1 = np.array(FFT, dtype = np.uint8) 

fig.add_subplot(223)
plt.imshow(FFT1, cmap="gray")
plt.title("FFT")

#defining the fraction of coefficients we need to keep, for each direction
keep_fraction=0.15
FFT2=FFT.copy()
r,c=FFT2.shape#taking the rows and columns

#setting to 0 all the r and c with indices between r,c*keep_fraction and r,c*(1-keep_fraction)
FFT2[int (r*keep_fraction):int(r*(1-keep_fraction))]=0
FFT2[:, int (c*keep_fraction):int(c*(1-keep_fraction))]=0

#taking the IFFT
FFT3=ifftn(FFT2).real
FFT3=np.array(FFT3, dtype=np.uint8)

fig.add_subplot(224)
plt.imshow(FFT3, cmap="gray")
plt.title("Denoised Image")