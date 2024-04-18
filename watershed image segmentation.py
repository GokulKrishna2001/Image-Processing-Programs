#watershed image segmentation

import numpy as np
import cv2
from matplotlib import pyplot as plt
fig=plt.figure(figsize=(15,15))

image=cv2.imread('images/walkbridge.tif')
img1=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
img2=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#converting to the respective rgb channels
b,g,r=cv2.split(img1)

#finding the otsu's threshold of the image
_,thresh=cv2.threshold(img2,150,255,cv2.THRESH_BINARY,cv2.THRESH_OTSU)

thresh1=cv2.cvtColor(thresh,cv2.COLOR_BGR2RGB)
fig.add_subplot(331)
plt.imshow(img1)
plt.title("Original Image")

fig.add_subplot(332)
plt.imshow(thresh1)
plt.title("Image after Otsu's Threshold")

#finding the kernel
kernel=np.ones((2,2),np.uint8)
#performing the morphological operation to remove noise and fill in the gaps
#cv2.MORPH_CLOSE: for smoothing contours, deleting small holes, and joining closely spaced objects
closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel, iterations = 2)

fig.add_subplot(334)
plt.imshow(closing,'gray')
plt.title('Image after morphological closing')

#finding the sure background area
sure_bg=cv2.dilate(closing,kernel,iterations=3)

fig.add_subplot(335)
plt.imshow(sure_bg,'gray')
plt.title('Sure Background of the Image')

#finding the sure foreground area
#function used to calculate the smallest distance from 0
#cv2.DIST_L2: euclidean distance
dist_transform=cv2.distanceTransform(sure_bg,cv2.DIST_L2,3)

fig.add_subplot(336)
plt.imshow(dist_transform,'gray')
plt.title('Distance Transform of the Image')

#finding the threshold of the above
_,sure_fg=cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)

fig.add_subplot(337)
plt.imshow(sure_fg,'gray')
plt.title('Sure Foreground of the Image')

#finding the unknown regions
sure_fg=np.uint8(sure_fg)
unknown_region=cv2.subtract(sure_bg,sure_fg)

fig.add_subplot(338)
plt.imshow(unknown_region,'gray')
plt.title('Unknown Regions of the Image')

#labelling the markers for water shed
_,markers=cv2.connectedComponents(sure_fg)
#adding 1 to all the labels so that the sure bg is not 0 but 1
markers=markers+1
#marking the unknown regions with 0
markers[unknown_region==255]=0
markers=cv2.watershed(img1,markers)
#marking the watershed boundaries
img1[markers==-1]=[255,0,0]

fig.add_subplot(339)
plt.imshow(img1)
plt.title("Original Image after watershed")
