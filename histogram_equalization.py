# -*- coding: utf-8 -*-
"""Histogram_equalization.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1MjMJekRc-GkFg95u6i4qrk98RxNvMfPt

# Task-A
"""

from matplotlib import pyplot as plt
import numpy as np
import cv2

img = cv2.imread("Fig0310(a)(Moon Phobos).tif",0)
plt.imshow(img, cmap='gray')
plt.show()

hist,bins = np.histogram(img.flatten(),256,[0,256])
plt.hist(img.flatten(),256,[0,256],color='r')
plt.xlim([0,256])
plt.xlabel('Intensity')
plt.ylabel('Pixel count')
plt.show()

'''
def Histogram_Equalization(image):
    grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_img = cv2.equalizeHist(grayimg)
    return equalized_img

equalized_img = Histogram_Equalization(image)
'''

"""# Task-B"""

equalised_image=cv2.equalizeHist(img)

hist,bins = np.histogram(equalised_image.flatten(),256,[0,256])
plt.hist(equalised_image.flatten(),256,[0,256],color='r')

plt.xlim([0,256])
plt.xlabel('Intensity')
plt.ylabel('Pixel count')
plt.show()

plt.imshow(equalised_image, cmap='gray')
plt.show()

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(equalised_image, cmap='gray')

plt.show()

figure, axes = plt.subplots(nrows = 1, ncols = 2, dpi=100)
axes[0].set_title('Input Image')
axes[0].imshow(img, cmap='gray')

axes[1].set_title('Equalized Image')
axes[1].imshow(equalised_image, cmap='gray')

figure.tight_layout()
plt.show()

"""# Task-C"""

from pylab import concatenate, normal
from pylab import *

data=concatenate((normal(40,17,5000),normal(240,10,290)))
y,x,_=hist(data,100,alpha=0.4,label='data')

plt.close()

x=(x[1:]+x[:-1])/2 

def gauss(x,mu,sigma,A):
    return A*exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

expected = bimodal(x,40,17,5000,190,10,290)
new_exp = expected/290000
plt.plot(x,new_exp,color='blue')
plt.xlim(0, 255)
plt.ylim(-0.001, 0.02)

plt.title("Bimodal Gaussian Function--Specified Histogram")
plt.show(block=True)