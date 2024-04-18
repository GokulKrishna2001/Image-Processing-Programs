#log transformation

import cv2
from matplotlib import pyplot as plt
import numpy as np

#initializing the plot size ,rows and columns
fig = plt.figure(figsize=(10,7))

#reading the image and converting to rgb and displaying using plt.imshow
img1 = cv2.imread('images/peppers_color_256.tif')
image= cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)


#applying log transformation in an image ,S = c * log (1 + r)
#c is the maximum output value, therefore maximum output values wrt 255
#c = 255 / (log (1 + max_input_pixel_value))

c = 255 / np.log(1 + np.max(image))
log_image = c * (np.log(image + 1)) 

# Specify the data type so that 
# float value will be converted to int 
#dtype=data type 
#uint88=unassigned integer with 8 bits i.e; the max value is 11111111 which is 255
log_image = np.array(log_image, dtype = np.uint8) 

#plotting
fig.add_subplot(121)
plt.imshow(image)
plt.title('Original Image')

fig.add_subplot(122)
plt.imshow(log_image)
plt.title('Log Transformed Image')