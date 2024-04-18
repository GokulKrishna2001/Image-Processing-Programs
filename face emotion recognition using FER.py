#face emotion detection
#firstly,install this:
#!pip install fer

#part 2
#!pip install tensorflow

#part 3
#download a collage of faces with emotions from the net or a single face
from fer import FER
import cv2
from matplotlib import pyplot as plt
fig=plt.figure(figsize=(6,6))

image=cv2.imread("images/lena_color_256.tif")
img1=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

emotion_detector=FER(mtcnn=True)#Multi-task Cascaded Convolutional Networks
analysis=emotion_detector.detect_emotions(image)

fig.add_subplot(111)
plt.imshow(img1)
plt.title("Image")
print(type(analysis))
i=1
for person in analysis:
    print(f'Person #{i}:')
    print("Box:",person['box'])
    print("Emotions:",person['emotions'])
    print()
    i=i+1
