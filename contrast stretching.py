#contrast stretching

import cv2
from matplotlib import pyplot as plt

#dclaring the plot
fig=plt.figure(figsize=(10,5))

#input the image
img1=cv2.imread('images/lena_gray_256.tif')
image1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

#doing contrast stretching with the equation:
#g=((f-f.min())/(f.max()-f.min()))
image2=((image1-image1.min())/(image1.max()-image1.min()))

#adding subplots
fig.add_subplot(121)
plt.imshow(image1)
plt.title("Original Image")

fig.add_subplot(122)
plt.imshow(image2)
plt.title("Contrast Stretched Image")