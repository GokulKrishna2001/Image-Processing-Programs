#image super resolution
#using interpolation

import cv2
from matplotlib import pyplot as plt

fig=plt.figure(figsize=(8,8))
image1=cv2.imread('images/lena_color_64.tif')
img1=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)

scale=int(input("Enter the scaling factor:"))
new_image=cv2.resize(img1,None, fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)

fig.add_subplot(121)
plt.imshow(img1)
plt.title("Original Image")

fig.add_subplot(122)
plt.imshow(new_image)
plt.title("New Image")