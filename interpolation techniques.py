#interpolation

import cv2
from matplotlib import pyplot as plt

#declaring fig
fig=plt.figure(figsize=(10,7))

#taking the image as input
image1=cv2.imread('images/mandril_color_128.tif')
img1=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)

#interpolation techniques
#bilinear
img2=cv2.resize(img1,dsize=(256,256),interpolation=cv2.INTER_LINEAR)
#bicubic
img3=cv2.resize(img1,dsize=(512,512),interpolation=cv2.INTER_CUBIC)
#nearest neighbor
img4=cv2.resize(img1,dsize=(1024,1024),interpolation=cv2.INTER_NEAREST)


#plotting
fig.add_subplot(221)
plt.imshow(img1)
plt.title("Original")

fig.add_subplot(222)
plt.imshow(img2)
plt.title("Bilinear Interpolation")

fig.add_subplot(223)
plt.imshow(img3)
plt.title("BiCubic Interpolation")

fig.add_subplot(224)
plt.imshow(img4)
plt.title("Nearest Neighbor Interpolation")