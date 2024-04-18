#dark image enhancement

import cv2
from matplotlib import pyplot as plt
fig=plt.figure(figsize=(8,8))

image1=cv2.imread("images/lake.tif")

img1=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
img2=cv2.cvtColor(image1,cv2.COLOR_BGR2HSV)

#splitting to the hsv channels
hue,sat,value=cv2.split(img2)

#input the brightness_multiplier
brightness_multiplier=float(input("Enter the Brightness Multiplier(0,1,2,3):"))

# Increase the brightness, or Value channel
#cv2.addWeighted(src1, alpha, src2, beta, gamma)
#src1(x,y)×alpha+src2(x,y)×beta+gamma
new_value = cv2.addWeighted(value, brightness_multiplier, 0, 0, 0)

# Recombine channels and convert back to RGB
new_img = cv2.merge((hue, sat, new_value))
enhanced_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2RGB)

fig.add_subplot(121)
plt.imshow(img1)
plt.title("Input Image")

fig.add_subplot(122)
plt.imshow(enhanced_img)
plt.title("Dark Image Enhancement")
