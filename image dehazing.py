#image dehazing
#pip install image-dehazer
import image_dehazer
import cv2
from matplotlib import pyplot as plt
fig=plt.figure(figsize=(10,10))

image1=cv2.imread('images/hazed_image.png')
image1=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
Dehazed_image, HazeMap = image_dehazer.remove_haze(image1)

cv2.waitKey(0)
fig.add_subplot(131)
plt.imshow(image1)
plt.title("Original Image")

fig.add_subplot(132)
plt.imshow(Dehazed_image, cmap="gray")
plt.title("Dehazed Image")

fig.add_subplot(133)
plt.imshow(HazeMap, cmap="gray")
plt.title("Haze Map")