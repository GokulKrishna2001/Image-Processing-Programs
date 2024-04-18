#morphological operations on lena, coins, rice image
#operations: erosion, dilation, opening, closing

import cv2
import numpy as np
from matplotlib import pyplot as plt
fig=plt.figure(figsize=(20,20))

def morph_ops(image,i):
    img1=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    img2=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    #erosion
    #to erode the outer surface/foreground of the image

    _,binary_img=cv2.threshold(img2,100,255,cv2.THRESH_BINARY)
    kernel=np.ones((3,3),np.uint8)
    #invert the binary image
    invert_bin_img=cv2.bitwise_not(binary_img)

    erosion=cv2.erode(invert_bin_img,kernel,iterations=1)
    erosion=cv2.cvtColor(erosion,cv2.COLOR_BGR2RGB)

    #dilation
    #dilates the foreground of the image
    #expands the foreground of the image and make it white
    dilation=cv2.dilate(invert_bin_img,kernel,iterations=1)
    dilation=cv2.cvtColor(dilation,cv2.COLOR_BGR2RGB)

    #opening
    #involves erosion followed by dilation in the foreground of the image
    opening=cv2.morphologyEx(binary_img,cv2.MORPH_OPEN,kernel,iterations=1)
    opening=cv2.cvtColor(opening,cv2.COLOR_BGR2RGB)
    
    #closing
    #involves dilation followed by erosion in the outer surface of the image
    closing=cv2.morphologyEx(binary_img,cv2.MORPH_CLOSE,kernel,iterations=1)
    closing=cv2.cvtColor(closing,cv2.COLOR_BGR2RGB)

    fig.add_subplot(6,3,i)
    plt.imshow(img1)
    plt.title('Original Image')
    i=i+1
    fig.add_subplot(6,3,i)
    plt.imshow(binary_img,cmap="gray")
    plt.title('Binary Image')
    i=i+1
    fig.add_subplot(6,3,i)
    plt.imshow(erosion)
    plt.title("Image after Erosion")
    i=i+1
    fig.add_subplot(6,3,i)
    plt.imshow(dilation)
    plt.title("Image after Dilation")
    i=i+1
    fig.add_subplot(6,3,i)
    plt.imshow(opening)
    plt.title("Image after Opening")
    i=i+1
    fig.add_subplot(6,3,i)
    plt.imshow(closing)
    plt.title("Image after Closing")
    i=i+1
    
    return i

i=int(1)
image=cv2.imread("images/lena_color_256.tif")
i=morph_ops(image,i)

image=cv2.imread("images/coin_image2.jpg")
i=morph_ops(image,i)

image=cv2.imread("images/rice_image.png")
i=morph_ops(image,i)