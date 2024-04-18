#without using the threshold function, to convert the gray image to binary image

import cv2
import numpy as np
from matplotlib import pyplot as plt

#plotting
fig=plt.figure(figsize=(10,5))

threshold=int(input("Enter the Threshold Value:"))

#reading the image in gray
image1=cv2.imread('images/peppers_color.tif',0)
gray_img = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

#creating an arrays with 0s with the size of the gray image
binary_img=np.zeros_like(gray_img)

#going through each pixel in the gray image
#shape gives the width, height
#check if the pixel at gray image at that point is greater than the threshold given
#if greather than the value, convert it to gray's 255 for binary image
for i in range(gray_img.shape[0]):
    for j in range(gray_img.shape[1]):
        if image1[i,j]>threshold:
            binary_img[i,j]=255#giving to white

#plotting
fig.add_subplot(121)
plt.imshow(gray_img)
plt.title("Gray Image")

fig.add_subplot(122)
plt.imshow(binary_img)
plt.title("Binary Image")
