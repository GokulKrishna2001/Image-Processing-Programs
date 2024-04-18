#superpixels segmentation, SLIC
from skimage.segmentation import slic
from skimage.color import label2rgb
import cv2
from matplotlib import pyplot as plt
fig=plt.figure(figsize=(8,8))

image=cv2.imread('images/mandril_color.tif')
img1=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

#obtaining the segments
segments=slic(img1)

#puting the segments on top of the original image to compare
#average colors of the superpixels are taken
segmented_image=label2rgb(segments,img1,kind='avg')

fig.add_subplot(121)
plt.imshow(img1)
plt.title('Original Image')

fig.add_subplot(122)
plt.imshow(segmented_image)
plt.title('SLIC Image')