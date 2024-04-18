#Viola jones face detection
import cv2
from matplotlib import pyplot as plt
fig=plt.figure(figsize=(8,8))

image=cv2.imread('images/lena_color_256.tif')
img1=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
img2=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

fig.add_subplot(121)
plt.imshow(img2)
plt.title("Original Image")

#A cascade object usually consists of a series of weak classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#multiscale indicates that the algo looks at the subregions of the images in mulitple scales
#to detect faces of various sizes
#contains all the detections needed for the target image
detected_faces = face_cascade.detectMultiScale(img1)

#to draw the rectangular box over the image
#rectangle(image, coordinate of top-left of the detection, same of bottom right, RGB color value, thickness)
for(column, row, width, height) in detected_faces:
    cv2.rectangle(img2,(column,row),(column+width,row+height),(0,255,0),2)

fig.add_subplot(122)
plt.imshow(img2)
plt.title("Recognized Image")