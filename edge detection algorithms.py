#edge detection algorithms
#canny, sobel and kirsch compass
import cv2
import numpy as np
from matplotlib import pyplot as plt
fig=plt.figure(figsize=(25,25))

image=cv2.imread('images/jetplane.tif')
img1=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
img2=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#for canny edge detection
blurred_image=cv2.GaussianBlur(img1,ksize=(3,3),sigmaX=2)
edges=cv2.Canny(blurred_image,70,135)
#70,135 are the threshold values
#values below 70 are not edges
#values above 135 are edges
#values between them are edges only if it is connected to strong edges
img3=cv2.cvtColor(edges,cv2.COLOR_BGR2RGB)

#for sobel, calculate the x and y
#cv2.CV_64F is the floating point data type for better calculations
#1,0 means the derivative in the x axis
#0,1 means the derivative in the y axis
sobel_x_axis=cv2.Sobel(img2,cv2.CV_64F,1,0,ksize=3)
sobel_y_axis=cv2.Sobel(img2,cv2.CV_64F,0,1,ksize=3)

# Convert Sobel outputs to appropriate data type
#scaling the values to the different data types
sobel_x_axis = cv2.convertScaleAbs(sobel_x_axis)
sobel_y_axis = cv2.convertScaleAbs(sobel_y_axis)

img4=cv2.cvtColor(sobel_x_axis,cv2.COLOR_GRAY2BGR)
img5=cv2.cvtColor(sobel_y_axis,cv2.COLOR_GRAY2BGR)

#using kirsch compass method

#kernels for the 8 directions
east=np.array([[-3,-3,-3],[-3,0,-3],[5,5,5]])
west=np.array([[5,5,5],[-3,0,-3],[-3,-3,-3]])
north=np.array([[-3,-3,5],[-3,0,5],[-3,-3,5]])
south=np.array([[5,-3,-3],[5,0,-3],[5,-3,-3]])
north_east=np.array([[-3,-3,-3],[-3,0,5],[-3,5,5]])
north_west=np.array([[-3,5,5],[-3,0,5],[-3,-3,-3]])
south_west=np.array([[-3,-3,-3],[5,0,-3],[5,5,-3]])
south_east=np.array([[5,5,-3],[5,0,-3],[-3,-3,-3]])

#calculating the kirsch compass direction edges with the above kernels and the gray image
east_img=cv2.filter2D(img2,-1,east)#-1 is the depth
img6=cv2.cvtColor(east_img,cv2.COLOR_GRAY2BGR)

west_img=cv2.filter2D(img2,-1,west)
img7=cv2.cvtColor(west_img,cv2.COLOR_GRAY2BGR)

north_img=cv2.filter2D(img2,-1,north)
img8=cv2.cvtColor(north_img,cv2.COLOR_GRAY2BGR)

south_img=cv2.filter2D(img2,-1,south)
img9=cv2.cvtColor(south_img,cv2.COLOR_GRAY2BGR)

NE_img=cv2.filter2D(img2,-1,north_east)
img10=cv2.cvtColor(NE_img,cv2.COLOR_GRAY2BGR)

NW_img=cv2.filter2D(img2,-1,north_west)
img11=cv2.cvtColor(NW_img,cv2.COLOR_GRAY2BGR)

SW_img=cv2.filter2D(img2,-1,south_west)
img12=cv2.cvtColor(SW_img,cv2.COLOR_GRAY2BGR)

SE_img=cv2.filter2D(img2,-1,south_east)
img13=cv2.cvtColor(SE_img,cv2.COLOR_GRAY2BGR)


#plotting everything
fig.add_subplot(631)
plt.imshow(img1)
plt.title("Image")

fig.add_subplot(632)
plt.imshow(blurred_image)
plt.title("Blurred Image")

fig.add_subplot(633)
plt.imshow(img3)
plt.title("Image after Canny's Method")

fig.add_subplot(634)
plt.imshow(img4)
plt.title("Sobel in X-Axis")

fig.add_subplot(635)
plt.imshow(img5)
plt.title("Sobel in Y-Axis")

fig.add_subplot(637)
plt.imshow(img6)
plt.title("Kirsch Compass in East Direction")

fig.add_subplot(638)
plt.imshow(img7)
plt.title("Kirsch Compass in West Direction")

fig.add_subplot(639)
plt.imshow(img8)
plt.title("Kirsch Compass in North Direction")

fig.add_subplot(6,3,10)
plt.imshow(img9)
plt.title("Kirsch Compass in South Direction")

fig.add_subplot(6,3,11)
plt.imshow(img10)
plt.title("Kirsch Compass in North-East Direction")

fig.add_subplot(6,3,12)
plt.imshow(img11)
plt.title("Kirsch Compass in North-West Direction")

fig.add_subplot(6,3,13)
plt.imshow(img12)
plt.title("Kirsch Compass in South-West Direction")

fig.add_subplot(6,3,14)
plt.imshow(img13)
plt.title("Kirsch Compass in South-East Direction")