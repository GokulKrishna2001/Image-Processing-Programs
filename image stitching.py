#image stitching
#panaromic stitching

import cv2
from matplotlib import pyplot as plt
fig=plt.figure(figsize=(10,10))

image1=cv2.imread('images/pan1.png')
image2=cv2.imread('images/pan2.png')

image1=cv2.cvtColor(image1,cv2.COLOR_BGR2RGB)
image2=cv2.cvtColor(image2,cv2.COLOR_BGR2RGB)
fig.add_subplot(131)
plt.imshow(image1)
plt.title("Image Part 1")

fig.add_subplot(132)
plt.imshow(image2)
plt.title("Image Part 2")

image_list=[image1,image2]
stitching=cv2.Stitcher.create()
status, stitched_image=stitching.stitch(image_list)

#checking if the stitching was successful
if status != cv2.STITCHER_OK: 
    print("Stitching Issue")
    
fig.add_subplot(133)
plt.imshow(stitched_image)
plt.title("Stitched Image")