#image blurring and restoration working code

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import fftn, ifftn, fftshift
fig=plt.figure(figsize=(10,10))

#function for creating a gaussian filter mask
def gaussian_filter(k, sigma):
    arx = np.arange((-k // 2) + 1.0, (k // 2) + 1.0)
    x, y = np.meshgrid(arx, arx)
    filt = np.exp(-(1/2) * (np.square(x) + np.square(y)) / np.square(sigma))
    return filt / np.sum(filt)

image1=cv2.imread('images/peppers_color_256.tif')
img1=cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)

mask=gaussian_filter(k=9, sigma=2.5)

#computing the number of padding on one side
a=int(img1.shape[0]//2 - mask.shape[0]//2)
mask_pad=np.pad(mask, (a,a-1), 'constant', constant_values=(0))

IMG=fftn(img1)
M=fftn(mask_pad)

#convolution
G=np.multiply(IMG,M)

#inverse transform
g=fftshift(ifftn(G).real)

fig.add_subplot(131)
plt.imshow(img1, cmap="gray")
plt.title("Original Image")

fig.add_subplot(132)
plt.imshow(g, cmap="gray")
plt.title("Blurred Image")

#finding Image=G/Mask
IMG_inv=np.divide(G,M)
img1_restored=ifftn(IMG_inv).real

fig.add_subplot(133)
plt.imshow(img1_restored, cmap="gray")
plt.title("Restored Image")