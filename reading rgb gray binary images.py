#reading a rgb, grey and binary image
#pip install matplotlib
import cv2
from matplotlib import pyplot as plt

#creating the plot figure
fig=plt.figure(figsize=(10,7))
row=2
column=2

#reading 3 images
#open cv uses bgr format. so, we have to convert BGR2RGB
image1=cv2.imread('images/mandril_color.tif')
img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

#reading a gray image
img2=cv2.imread('images/mandril_gray.tif')
#img4=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

#reading an image for binary
img3=cv2.imread('images/mandril_gray.tif')
(threshold, binary_img5)=cv2.threshold(img3,150,255,cv2.THRESH_BINARY)

#plotting the images to the plot
fig.add_subplot(131)
plt.imshow(img1)
plt.title("RGB")

fig.add_subplot(132)
plt.imshow(img2)
plt.title("Gray Scale")

fig.add_subplot(133)
plt.imshow(binary_img5)
plt.title("Binary")