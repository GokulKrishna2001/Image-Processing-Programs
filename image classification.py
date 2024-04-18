#image classification of natural, satellite and medical image

import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import os

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

fig=plt.figure(figsize=(10,10))

#a set of 3 classes with 30 images each are given
path='dataset/'

#training dataset
train_ds= tf.keras.utils.image_dataset_from_directory(path, validation_split=0.2,subset="training",seed=123,image_size=(50,50),batch_size=30)
#validation dataset
val_ds=tf.keras.utils.image_dataset_from_directory(path, validation_split=0.2, subset="validation",seed=123, image_size=(50,50),batch_size=30)

#getting the class names and its number
class_names=train_ds.class_names
length=len(class_names)

#creating the model
model=Sequential([
    layers.Rescaling(1./255, input_shape=(50,50, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(length)
])

#compiling the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#training the model with 5 epochs
epoch=5
history=model.fit(train_ds, validation_data=val_ds,epochs=epoch)

#giving input 1
input1='images/medical.jpg'
img1=cv2.imread(input1)
img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
#loading the input path to keras
input1=tf.keras.utils.load_img(input1, target_size=(50,50))
#converting the input to array
img_arr1=tf.keras.utils.img_to_array(input1)
img_arr1=tf.expand_dims(img_arr1, 0)# creating a batch
#predicting the class from the model and finding the score
predictions=model.predict(img_arr1)
score=tf.nn.softmax(predictions[0])
#plotting the input image with the predicted class
fig.add_subplot(131)
plt.imshow(img1)
plt.title(f'Image from Class:{class_names[np.argmax(score)]}')


#input 2
input2='images/satellite.jpg'
img2=cv2.imread(input2)
img2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

input2=tf.keras.utils.load_img(input2, target_size=(50,50))

img_arr2=tf.keras.utils.img_to_array(input2)
img_arr2=tf.expand_dims(img_arr2, 0)# creating a batch

predictions=model.predict(img_arr2)
score=tf.nn.softmax(predictions[0])

fig.add_subplot(132)
plt.imshow(img2)
plt.title(f'Image from Class:{class_names[np.argmax(score)]}')


#input 3
input3='images/nature.jpg'
img3=cv2.imread(input3)
img3=cv2.cvtColor(img3,cv2.COLOR_BGR2RGB)

input3=tf.keras.utils.load_img(input3, target_size=(50,50))

img_arr3=tf.keras.utils.img_to_array(input3)
img_arr3=tf.expand_dims(img_arr3, 0)# creating a batch

predictions=model.predict(img_arr3)
score=tf.nn.softmax(predictions[0])

fig.add_subplot(133)
plt.imshow(img3)
plt.title(f'Image from Class:{class_names[np.argmax(score)]}')