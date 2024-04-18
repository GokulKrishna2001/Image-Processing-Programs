#bit plane slicing
import cv2
import numpy as np
from matplotlib import pyplot as plt
fig=plt.figure(figsize=(10,10))

image=cv2.imread('images/jetplane.tif')
image1=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

fig.add_subplot(331)
plt.imshow(gray,cmap="gray")
plt.title("Original Gray Image")

#creating the binary 8 bit representation of the array
arr=[]
for i in range(gray.shape[0]):
    for j in range(gray.shape[1]):
        arr.append(np.binary_repr(gray[i][j],width=8))#8 bits
gray_arr=np.array(arr)
gray_arr=gray_arr.reshape(gray.shape)

#creating the bit planes
bit_plane=np.zeros(gray.shape)


for plane in range(0,8):
    for i in range(gray_arr.shape[0]):
        for j in range(gray_arr.shape[1]):
            bit_plane[i][j]=int(gray_arr[i,j][plane])
    bit_plane=bit_plane*255
    fig.add_subplot(3,3,plane+2)
    plt.imshow(bit_plane,cmap="gray")
    plt.title(f'Bit plane {7-plane}')