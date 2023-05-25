# -*- coding: utf-8 -*-
"""
Created on Wed May 24 21:10:20 2023

@author: monaf
"""
import cv2 
import matplotlib.pyplot as plt 
import numpy as np
import math

img=cv2.imread("monalisa.jpg")
def Nearest_Neighbor(img):
    height,width,channels =img.shape
    emptyImage=np.zeros((1024,1024,channels),np.uint8)
    sh=1024/height
    sw=1024/width
    for i in range(1024):
        for j in range(1024):
            x=int(i/sh)
            y=int(j/sw)
            emptyImage[i,j]=img[x,y]
    return emptyImage

histogram = cv2.calcHist([img], [0], None, [256], [0, 256])

# Plot the histogram
plt.plot(histogram, color='blue')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()



