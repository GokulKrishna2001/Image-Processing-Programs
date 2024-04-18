#image denoising/ noise removal using median filter

import cv2 
import numpy as np 
from matplotlib import pyplot as plt

fig=plt.figure(figsize=(8,8))
image1= cv2.imread('images/lena_gray_256.tif',cv2.IMREAD_GRAYSCALE) 

#binary image conversion
binary_image = np.zeros_like(image1) 
# Obtain the number of rows and columns of the image
rows= image1.shape[0]
columns=image1.shape[1]

#let threshold=150
threshold=150
for i in range(rows):
    for j in range(columns):
        if image1[i,j]> threshold:
            binary_image[i,j]=255
        else:
            binary_image[i,j]=0

# Traverse the image. For every 3X3 area,  
# find the median of the pixels and 
# replace the center pixel by the median 
img1 = np.zeros([rows, columns]) 
  
for i in range(1, rows-1): 
    for j in range(1, columns-1): 
        temp = np.array([binary_image[i-1, j-1], binary_image[i-1, j], binary_image[i-1, j + 1], binary_image[i, j-1], binary_image[i, j], binary_image[i, j + 1], binary_image[i + 1, j-1], binary_image[i + 1, j], binary_image[i + 1, j + 1]] )
        temp = sorted(temp)
        img1[i, j]= temp[4] 
        
img1 = img1.astype(np.uint8) 
#cv2.imwrite('new_median_filtered.png', img_new1)

img2=cv2.cvtColor(binary_image,cv2.COLOR_BGR2RGB)
img3=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

fig.add_subplot(121)
plt.imshow(img2)
plt.title('Original Binary Image')

fig.add_subplot(122)
plt.imshow(img3)
plt.title('Denoised Image')