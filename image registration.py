#Image registration
import cv2 
import numpy as np 
from matplotlib import pyplot as plt
  
# Open the image files. 
img1_color = cv2.imread("images/lena_color_rotated2.tif")  # Image to be aligned. 
img2_color = cv2.imread("images/lena_color_256.tif")    # Reference image. 

# Convert to grayscale. 
img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY) 
img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY) 

height, width = img2.shape 
  
# Create ORB detector with 5000 features. 
orb_detector = cv2.ORB_create(5000) 
  
# Find keypoints and descriptors. 
# The first arg is the image, second arg is the mask 
#  (which is not required in this case). 
kp1, d1 = orb_detector.detectAndCompute(img1, None) 
kp2, d2 = orb_detector.detectAndCompute(img2, None) 
  
# Match features between the two images. 
# We create a Brute Force matcher with  
# Hamming distance as measurement mode. 
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True) 
  
# Match the two sets of descriptors. 
matches = matcher.match(d1, d2) 

# Select top N matches
N = 200
matches = matches[:N]

# Draw matched keypoints
img_matches = cv2.drawMatches(img2_color, kp1, img1_color, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display result
cv2.imshow('Matches', img_matches)
cv2.waitKey(0)
  
# Sort matches on the basis of their Hamming distance. 
matches=sorted(matches,key = lambda x: x.distance) 
  
# Take the top 90 % matches forward. 
matches = matches[:int(len(matches)*0.9)] 
no_of_matches = len(matches) 
  
# Define empty matrices of shape no_of_matches * 2. 
p1 = np.zeros((no_of_matches, 2)) 
p2 = np.zeros((no_of_matches, 2)) 
  
for i in range(len(matches)): 
    p1[i, :] = kp1[matches[i].queryIdx].pt 
    p2[i, :] = kp2[matches[i].trainIdx].pt 

# Find the homography matrix. 
homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC) 
  
# Use this matrix to transform the 
# colored image wrt the reference image. 
transformed_img = cv2.warpPerspective(img1_color, 
                    homography, (width, height)) 
  
# Save the output. 
cv2.imwrite('images/lena_color_256_new.tif', transformed_img) 

image1=cv2.cvtColor(img2_color,cv2.COLOR_BGR2RGB)
image2=cv2.cvtColor(img1_color,cv2.COLOR_BGR2RGB)
image3=cv2.cvtColor(transformed_img,cv2.COLOR_BGR2RGB)
fig=plt.figure(figsize=(15,15))
fig.add_subplot(131)
plt.imshow(image1)
plt.title("Original Image")

fig.add_subplot(132)
plt.imshow(image2)
plt.title("Rotated Image")

fig.add_subplot(133)
plt.imshow(image3)
plt.title("Corrected Image")
