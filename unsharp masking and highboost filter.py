#image sharpening
#unsharp masking and highboost filter

import cv2
import numpy as np
from matplotlib import pyplot as plt
fig=plt.figure(figsize=(15,15))

image=cv2.imread('images/lena_gray_256.tif')
image1=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

fig.add_subplot(141)
plt.imshow(image1)
plt.title("Original Image")

#blurred image
blurred_image=cv2.blur(image1,(5,5))#9,9 for pirate

fig.add_subplot(142)
plt.imshow(blurred_image)
plt.title("Blurred Image")

#for unsharp masking:
#sharpened image=original+(original-blurred)

unsharp_mask_image=image1+(image1-blurred_image)

fig.add_subplot(143)
plt.imshow(unsharp_mask_image)
plt.title("Image after Unsharp Msking")

#for highboost filtering
#Highboost=(A-1)original+(original-blurred)
#A=amount=1, HPF
#A>1, original image is added back to the highpass filter

A=2
highboost_image=(A-1)*image1+(image1-blurred_image)

fig.add_subplot(144)
plt.imshow(highboost_image)
plt.title("Image After High Boost Filter")