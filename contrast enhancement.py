#image contrast enhancement

#Import the necessary libraries 
import cv2 
import numpy as np 
from matplotlib import pyplot as plt

fig=plt.figure(figsize=(10,10))
# Load the image 
image= cv2.imread('images/peppers_color.tif') 
image1=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

#Plot the original image 
fig.add_subplot(121) 
plt.title("Original Image") 
plt.imshow(image1) 
  
# Adjust the brightness and contrast 
# Adjusts the brightness by adding the input to each pixel value 
brightness = int(input("Enter Brightness 1-10:"))
# Adjusts the contrast by scaling the input with the pixel value
contrast = int(input("Enter Contrast 1-10:"))
image2 = cv2.addWeighted(image1, contrast, np.zeros(image1.shape, image1.dtype), 0, brightness) 
  
#Save the image 
#Plot the contrast image 
fig.add_subplot(122) 
plt.title("Contrast Enhanced Image") 
plt.imshow(image2) 