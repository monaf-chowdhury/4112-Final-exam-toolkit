# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 23:39:27 2023

@author: monaf
"""

import cv2
import matplotlib.pyplot as plt 
import math
import numpy as np

img = cv2.imread('two_cats.jpg',cv2.IMREAD_GRAYSCALE) 
print("Original Image:")
plt.imshow(img,cmap='gray') 
cv2.imwrite('gray_img.jpg',img)
plt.show()

# Step 1: Apply Gaussian blur to the image
blur = cv2.GaussianBlur(img, (5, 5), 0)
final = cv2.Canny(blur, 100, 200)
print("\nGaussian smoothed image")
plt.imshow(blur,cmap='gray')
cv2.imwrite('Gaussian smoothing.jpg',blur)
plt.show()

# Step 2: Calculate the gradient magnitude and direction using Sobel operator
sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
grad_mag = np.sqrt(np.square(sobelx) + np.square(sobely))
grad_dir = np.arctan2(sobely, sobelx) * 180 / np.pi

# Step 3: Apply non-maximum suppression to the gradient magnitude image
suppressed = np.zeros_like(grad_mag)
angle = grad_dir + 180
for i in range(1, grad_mag.shape[0]-1):
    for j in range(1, grad_mag.shape[1]-1):
        q = 0
        r = 0
        
        if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
            q = grad_mag[i, j+1]
            r = grad_mag[i, j-1]
        elif 22.5 <= angle[i,j] < 67.5:
            q = grad_mag[i+1, j+1]
            r = grad_mag[i-1, j-1]
        elif 67.5 <= angle[i,j] < 112.5:
            q = grad_mag[i+1, j]
            r = grad_mag[i-1, j]
        elif 112.5 <= angle[i,j] < 157.5:
            q = grad_mag[i-1, j+1]
            r = grad_mag[i+1, j-1]
        if grad_mag[i,j] >= q and grad_mag[i,j] >= r:
            suppressed[i,j] = grad_mag[i,j]

print("\nNon maximum suppression")
plt.imshow(suppressed,cmap='gray')
cv2.imwrite('non maximum suppression.jpg',suppressed)
plt.show()


# Step 4: Define thresholds for hysteresis edge tracking
high_thresh = np.max(suppressed) * 0.2
low_thresh = high_thresh * 0.5

edges = np.zeros(suppressed.shape)
strong_edges_row, strong_edges_col = np.where(suppressed > high_thresh)
weak_i, weak_j = np.where((suppressed >= low_thresh) & (suppressed <= high_thresh))
edges[strong_edges_row, strong_edges_col] = 255
edges[weak_i, weak_j] = 50


print("\nDouble threshold")
plt.imshow(edges,cmap='gray')
cv2.imwrite('Double thresholding.jpg',edges)
plt.show()

# Step 5: Perform edge tracking by hysteresis
for i, j in zip(weak_i, weak_j):
    if any(edges[i-1:i+2, j-1:j+2].ravel()):
        edges[i, j] = 1

# Step 6: Show the final edge image

print("\nCanny edge detection")
plt.imshow(edges,cmap='gray')
cv2.imwrite('canny edge detection.jpg',edges)
plt.show()


img = cv2.imread('two_cats.jpg') 
def canny_edge_detection(image, sigma=0.33):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Compute the median of the blurred image
    median = np.median(blurred)
    
    # Calculate lower and upper thresholds using the median
    lower_threshold = int(max(0, (1.0 - sigma) * median))
    upper_threshold = int(min(255, (1.0 + sigma) * median))
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, lower_threshold, upper_threshold)
    
    return edges

plt.imshow(img,cmap='gray')

# Perform Canny edge detection
edges = canny_edge_detection(img)

# Display the original image and the detected edges
plt.imshow( img)
plt.show()
plt.imshow(edges)
plt.show()
#cv2.waitKey(0)
#cv2.destroyAllWindows()


img = cv2.imread('two_cats.jpg') 

def sobel_edge_detection(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Define Sobel masks
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    # Perform convolution using Sobel masks
    gradient_x = cv2.filter2D(blurred, -1, sobel_x)
    gradient_y = cv2.filter2D(blurred, -1, sobel_y)
    
    # Calculate the magnitude and direction of gradients
    gradient_magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    gradient_direction = np.arctan2(gradient_y, gradient_x)
    
    # Normalize the magnitude to the range [0, 255]
    gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))
    
    return gradient_magnitude, gradient_direction

# Perform Sobel edge detection
edges, directions = sobel_edge_detection(img)

plt.imshow( img)
plt.show()
plt.imshow(edges)
plt.show()
'''
# Display the original image and the detected edges
cv2.imshow('Original Image', image)
cv2.imshow('Sobel Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''








