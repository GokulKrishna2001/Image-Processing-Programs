#Image thresholding
#Otsu's thresholding
import cv2
from matplotlib import pyplot as plt

fig=plt.figure(figsize=(12,12))
image1 = cv2.imread('images/peppers_color_256.tif') 
  
img1=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY) 
  
threshold=int(input("Enter the threshold value:"))
_,img_reg_threshold = cv2.threshold(img2, threshold, 255, cv2.THRESH_BINARY)      
_,img_otsu = cv2.threshold(img2, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)      
  
img_regthresh=cv2.cvtColor(img_reg_threshold,cv2.COLOR_BGR2RGB)
img_otsu=cv2.cvtColor(img_otsu,cv2.COLOR_BGR2RGB)

fig.add_subplot(131)
plt.imshow(img1)
plt.title("Input Image")

fig.add_subplot(132)
plt.imshow(img_regthresh)
plt.title("Image after Binary Thresholding")

fig.add_subplot(133)
plt.imshow(img_otsu)
plt.title("Image after Otsu's Thresholding")