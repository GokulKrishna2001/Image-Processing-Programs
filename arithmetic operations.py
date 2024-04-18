#arithmetic operations

import cv2
from matplotlib import pyplot as plt

fig=plt.figure(figsize=(15,15))

img1=cv2.imread("images/mandril_color.tif")
image1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

img2=cv2.imread("images/lena_color_512.tif")
image2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

#adding 2 images
image3=cv2.add(image1,image2)

#subtracting 2 images
image4=cv2.subtract(image1,image2)

#multiplying 2 images
image5=cv2.multiply(image1,image2)

#dividing 2 images
image6=cv2.divide(image1,image2)

#scalar multiplication and division
image11=cv2.multiply(image1,2)
image12=cv2.divide(image1,2)

fig.add_subplot(421)
plt.imshow(image1)
plt.title("Image1")

fig.add_subplot(422)
plt.imshow(image2)
plt.title("Image 2")

fig.add_subplot(423)
plt.imshow(image3)
plt.title("After Addition")

fig.add_subplot(424)
plt.imshow(image4)
plt.title("After Subtraction")

fig.add_subplot(425)
plt.imshow(image5)
plt.title("After Multiplication")

fig.add_subplot(426)
plt.imshow(image6)
plt.title("After Division")

fig.add_subplot(427)
plt.imshow(image11)
plt.title("After Scalar Multiplication of Image1")

fig.add_subplot(428)
plt.imshow(image12)
plt.title("After Scalar Division of Image1")