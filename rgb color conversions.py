#all outputs are in gray
#color conversions
#RGB to YCbCr, to La*b* and HSI
import cv2
import math
from matplotlib import pyplot as plt
import numpy as np
fig=plt.figure(figsize=(12,12))

image=cv2.imread('images/peppers_color_256.tif')
image1=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image_ycbcr=cv2.cvtColor(image1,cv2.COLOR_RGB2YCrCb)#RGB to YCrCb
image_lab=cv2.cvtColor(image1,cv2.COLOR_RGB2LAB)#RGB to La*b*
arr1=np.asarray(image_ycbcr)
arr2=np.asarray(image_lab)


l_channel, a_channel, b_channel = cv2.split(image_lab)
fig.add_subplot(331)
plt.imshow(arr1[:,:,0],cmap="gray")
plt.title("Luminance Y")

fig.add_subplot(332)
plt.imshow(arr1[:,:,1],cmap="gray")
plt.title("Red Difference Chrominance Cr")

fig.add_subplot(333)
plt.imshow(arr1[:,:,2],cmap="gray")
plt.title("Blue Difference Chrominance Cb")

fig.add_subplot(334)
plt.imshow(l_channel,cmap="gray")
plt.title("Lightness L")

fig.add_subplot(335)
plt.imshow(a_channel,cmap="gray")
plt.title("Chroma on Green to Red axis: a*")

fig.add_subplot(336)
plt.imshow(b_channel,cmap="gray")
plt.title("Chroma on Blue to Yellow axis: b*")


#converting to HSI
rgb_array=np.float32(image1)/255
blue=rgb_array[:,:,0]
green=rgb_array[:,:,1]
red=rgb_array[:,:,2]

#calculating Hue
h_hsi=np.copy(red)

for i in range(0,blue.shape[0]):
    for j in range(0,blue.shape[0]):
        h_hsi[i][j]= 0.5 * ((red[i][j] - green[i][j]) + (red[i][j] - blue[i][j])) / math.sqrt((red[i][j] - green[i][j])**2 +((red[i][j] - blue[i][j]) * (green[i][j] - blue[i][j])))
        h_hsi[i][j]=math.acos(h_hsi[i][j])
        
        if blue[i][j]<=green[i][j]:
            h_hsi[i][j]=h_hsi[i][j]
        else:
            h_hsi[i][j]=((360*math.pi)/180.0)-h_hsi[i][j]

fig.add_subplot(337)
plt.imshow(h_hsi,cmap="gray")
plt.title("HSI image: Hue H")

#calculating saturation
min=np.minimum(np.minimum(red,green),blue)
s_hsi=1-(3/(red+blue+green+0.001)*min)

fig.add_subplot(338)
plt.imshow(s_hsi,cmap="gray")
plt.title("HSI image: Saturation S")

#calculating intensity:
i_hsi=np.divide(blue+green+red,3)
fig.add_subplot(339)
plt.imshow(i_hsi,cmap="gray")
plt.title("HSI image: Intensity I")
