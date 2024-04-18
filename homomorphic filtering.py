#code is not accurate as an enhancement is used in the end
#does give out different outputs
#homomorphic filtering
#steps:
#1) taking natural log
#2) FFT
#3) doing H(u,v) like a gaussian filter or so
#4) IFFT
#5) inverse log

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import fftn, ifftn, fftshift, ifftshift
fig=plt.figure(figsize=(12,12))

image=cv2.imread('images/homo.png')
image1=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

fig.add_subplot(231)
plt.imshow(gray,cmap="gray")
plt.title("Original Image")

#step1 1) taking natural log
c = 255 / np.log(1 + np.max(gray)) 
log_image = c * (np.log(gray + 1+1e-8)) 
# float value will be converted to int 
log_image = np.array(log_image, dtype = np.uint8) 

fig.add_subplot(232)
plt.imshow(log_image,cmap="gray")
plt.title("Log Image")

#step 2) taking FFT
IMG= fftn(log_image)
IMG1=np.array(IMG.real, dtype=np.uint8)

fig.add_subplot(233)
plt.imshow(IMG1,cmap="gray")
plt.title("FFT of the image")

#step 3) apply gaussian filter mask
def gaussian_filter(k, sigma):
    arx = np.arange((-k // 2) + 1.0, (k // 2) + 1.0)
    x, y = np.meshgrid(arx, arx)
    filt = np.exp(-(1/2) * (np.square(x) + np.square(y)) / np.square(sigma))
    return filt / np.sum(filt)

mask=gaussian_filter(k=9, sigma=3)
lp_mask=0.5*gaussian_filter(k=9,sigma=3)

#computing the number of padding on one side
a=int(gray.shape[0]//2 - mask.shape[0]//2)
mask_pad=np.pad(lp_mask, (a,a-1), 'constant', constant_values=(0))
mask_pad=1-mask_pad #creating a HPF
M=fftn(mask_pad)

#convolution
G=np.multiply(IMG,M)
g=fftshift(ifftn(G).real)

fig.add_subplot(234)
plt.imshow(g,cmap="gray")
plt.title("Gaussian Blur")

#step 4) taking IFFT
inv=np.divide(G,M)
inv=ifftn(inv).real
inv=np.array(inv, dtype=np.uint8)

fig.add_subplot(235)
plt.imshow(inv,cmap="gray")
plt.title("IFFT")

#step 5) Taking the exponent
inv_log_image=np.abs(np.exp(inv/c))
inv_log_image = np.array(inv_log_image, dtype = np.uint8) 

inv_log_image = cv2.addWeighted(inv_log_image, 1.6, np.zeros(gray.shape, gray.dtype), 0, 1) 
fig.add_subplot(236)
plt.imshow(inv_log_image,cmap="gray")
plt.title("Inverse Log Image")