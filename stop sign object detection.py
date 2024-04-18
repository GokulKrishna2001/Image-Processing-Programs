#object detection
#object detection of a stop sign
#download the XML file from GFG and download any stopsign image from the net and run
import cv2
from matplotlib import pyplot as plt

fig=plt.figure(figsize=(8,8))

image=cv2.imread("images/stop_sign.jpg")

img1=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
img11=img1
img2=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

fig.add_subplot(121)
plt.imshow(img1)
plt.title("Original Image")

#here we are importing the xml file for detecting a stop sign
#minSize prevents the false detections under that specific size
xml_data=cv2.CascadeClassifier('stop_data.xml')
found=xml_data.detectMultiScale(img2,minSize=(25,25))

#checking if the stop sign was found
len_found=len(found)
if len_found!=0: #stop sign exists
    for(column, row, width, height) in found:
        cv2.rectangle(img11,(column,row),(column+width,row+height),(0,255,0),4)

fig.add_subplot(122)
plt.imshow(img11)
plt.title('Object Detected')