#RGB image to gray scale

import cv2
from matplotlib import pyplot as plt

#declaring fig
fig=plt.figure(figsize=(10,5))

#taking the image as the input
image1=cv2.imread('images/lena_color_256.tif')
img1=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)

#plotting the original image
fig.add_subplot(121)
plt.imshow(img1)
plt.title("Color Image")

#getting the row and column size of img1
(row,column)=img1.shape[0:2]

#(R+G+B)/3
for i in range(row):
    for j in range(column):
        img1[i,j]=sum(img1[i,j])*0.33

#plotting the gray image
fig.add_subplot(122)
plt.imshow(img1)
plt.title("Gray Image")