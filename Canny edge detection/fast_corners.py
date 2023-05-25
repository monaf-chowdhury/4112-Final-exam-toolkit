# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 23:45:22 2023

@author: monaf
"""

import cv2
import matplotlib.pyplot as plt 
import math
import numpy as np

img = cv2.imread('fight_club.jpg',cv2.IMREAD_GRAYSCALE) 
print("Original Image:")
plt.imshow(img,cmap='gray') 
plt.show()

fast = cv2.FastFeatureDetector_create(threshold=50)  
keypoints = fast.detect(img, None) 
img_with_corners = cv2.drawKeypoints(img, keypoints, None, color=(0, 0, 255)) 

 
cv2.imwrite("FASTcorners.jpg", img_with_corners) 
plt.imshow(img_with_corners,cmap='gray')
plt.show()
