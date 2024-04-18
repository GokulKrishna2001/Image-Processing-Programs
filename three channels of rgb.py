#display the 3 channels of rgb of an image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open("images/peppers_color_256.tif")
M = np.asarray(img)

plt.figure(figsize=(12, 6))

plt.subplot(131)
plt.imshow(M[:, :, 0], cmap='Reds', vmin=0, vmax=255)
plt.title("Red Channel")

plt.subplot(132)
plt.imshow(M[:, :, 1], cmap='Greens', vmin=0, vmax=255)
plt.title("Green Channel")

plt.subplot(133)
plt.imshow(M[:, :, 2], cmap='Blues', vmin=0, vmax=255)
plt.title("Blue Channel")