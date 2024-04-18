#histogram equalization
import cv2
import numpy as np
from matplotlib import pyplot as plt

fig=plt.figure(figsize=(8,8))
# Reading the image
img = cv2.imread("images/lena_gray_256.tif",0)
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#img_1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

# original image
plt.subplot(221)
plt.title('Original Image')
plt.imshow(img1)
    
#histogram for the image
img_hist = cv2.calcHist([img],[0],None,[256],[0,256])
fig.add_subplot(222)
plt.title('Histogram of the Image')
plt.plot(img_hist)

#for the equalized image
new_image=cv2.equalizeHist(img)
fig.add_subplot(223)
plt.title('Image after Histogram Equalization')
plt.imshow(new_image, cmap='gray')
#cmap=gray is used to specify that the image is displayed in grayscale

#for the equalized histogram
img_hist_new = cv2.calcHist([new_image],[0],None,[256],[0,256])
fig.add_subplot(224)
plt.title('Equalized Histogram')
plt.plot(img_hist_new)