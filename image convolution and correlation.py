#convolution and correlation of the images
import cv2
import numpy as np
from matplotlib import pyplot as plt
fig=plt.figure(figsize=(12,12))

image1=cv2.imread('images/lena_color_256.tif')
image2=cv2.imread('images/peppers_color.tif')
img1=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
img2=cv2.cvtColor(image2,cv2.COLOR_BGR2RGB)

#applying box blur to perform convolution of image 1

#kernel size
ks=3
kernel=np.ones((ks,ks),np.float32)/(ks*ks)
Box_blur_image1=cv2.filter2D(image1,-1,kernel)
Box_blur_image1=cv2.cvtColor(Box_blur_image1,cv2.COLOR_BGR2RGB)
#applying the gaussian blur to perform covolution of image 2
#Box_blur_image2=cv2.GaussianBlur(image2,ksize=(5,5),sigmaX=5,sigmaY=5)
#Box_blur_image2=cv2.cvtColor(Box_blur_image2,cv2.COLOR_BGR2RGB)

#correlation=correlate2d(gray1, gray2, mode='same', boundary='fill', fillvalue=0)

correlation=cv2.matchTemplate(img1,img2,cv2.TM_CCORR_NORMED)

fig.add_subplot(221)
plt.imshow(img1)
plt.title('Image 1')

fig.add_subplot(222)
plt.imshow(Box_blur_image1)
plt.title('Image 1 after Box Blur Convolution')

fig.add_subplot(223)
plt.imshow(img2)
plt.title("Image 2")

#fig.add_subplot(235)
#plt.imshow(Box_blur_image2)
#plt.title("Image 2 after Gaussian Blur Convolution")

fig.add_subplot(224)
plt.imshow(correlation)
plt.title("Correlation of the 2 Images")