#image sampling and quantization
import cv2
from matplotlib import pyplot as plt
import numpy as np
fig=plt.figure(figsize=(8,8))

image=cv2.imread('images/mandril_color.tif')
image1=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

fig.add_subplot(131)
plt.imshow(gray,cmap="gray")
plt.title("Original image")

#taking a sampling factor
#taking the 6th row and 6th column alternatively to display
sampling_factor=6
sampled_image=gray[::sampling_factor,::sampling_factor]

#quantization
quantization_level=10
quantized_image=np.floor_divide(gray,256//quantization_level)*(256//quantization_level)

fig.add_subplot(132)
plt.imshow(sampled_image,cmap="gray")
plt.title("Sampled Image")

fig.add_subplot(133)
plt.imshow(quantized_image,cmap="gray")
plt.title("Quanitzed Image")