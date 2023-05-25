# -*- coding: utf-8 -*-
"""
Created on Wed May 24 21:24:31 2023

@author: monaf
"""
import cv2 
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('monalisa.jpg')
intensity_transformation_img = cv2.imread('input.png',cv2.COLOR_BGR2GRAY)
hist_eq = cv2.imread("Fig0310(a)(Moon Phobos).tif",cv2.IMREAD_GRAYSCALE)
hist_match = cv2.imread("aspens_in_fall.jpg")
hist_desire = cv2.imread("forest-resized.jpg")

############################ Plotting ######################################
def histogram_plotting(img):
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    plt.plot(hist,color='blue')
    plt.xlabel('Intensity vlues')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.show()

def histogram_plotting_better(img):
    hist,bins= np.histogram(img.flatten(),256,[0,256])
    plt.hist(img.flatten(),256,[0,256],color='b')
    plt.xlim([0,256])
    plt.xlabel('Intensity')
    plt.ylabel('Pixel count')
    plt.title('Histogram')
    plt.show()
    
    
def plotting_function(img):
    plt.imshow(img)
    plt.show()
    
def Gamma_Transformation_plotting(img):
    gamma_vals = [0.1,0.5,1.0,2.0,3.0,4.0]
    gamma_corrected = Gamma_Transformation(img, gamma_vals)
    for i in range(len(gamma_vals)):
        print(f"\nThe following is the gamma transformed image with gamma value {gamma_vals[i]}: ")
        plt.imshow(gamma_corrected[i]) 
        plt.show() 

######################### Inter-polation ########################
def Nearest_Neighbour(img):
    height, width, channels = img.shape
    sh = 1024/height
    sw = 1024/ width
    empty = np.zeros((1024,1024,channels),np.uint8)  # Generating empty image to put the new img
    
    for i in range(1024):
        for j in range(1024):
            x = int(i /sh)
            y = int(j /sw)
            empty[i,j] = img [x,y]
    return empty


####################### Image intensity transoformation ##################

def Photgraphic_negative(img):
    # simple method: return 1-img
    return 1-img

def Log_transformation(img):
    c = 255 / np.log(1+np.max(img))
    logged_img = (c* (np.log(1+img))).astype(dtype = np.uint8)
    return logged_img

def Gamma_Transformation(img,gamma_vals):
    gamma_corrected = []
    for i in range(len(gamma_vals)):
        gamma_corrected.append(np.array(255* (img/255)**gamma_vals[i], dtype = np.uint8 ))
    return gamma_corrected

'''
def contrast_stretching(img, low, high):
    normalized_img = img.astype(float)/255.0
    stretch_img = (normalized_img-low) * (1/(high - low ))
    stretch_img = np.clip(stretch_img,0,1)
    stretch_img = (stretch_img*255).astype(dtype = np.uint8)
    return stretch_img 
'''
def Contrast_Stretching(image):
    stretched_image = image.copy()
    height, width, _ = image.shape 
    maxiI = 255
    miniI = 0
    maxoI = 100
    minoI = 0
    
    for i in range(0, height - 1): 
        for j in range(0, width - 1):  
            pixel = stretched_image[i, j] 
            '''
            pout = (pin - miniI) * ((maxoI-minoI) / (maxiI-miniI)) + minoI         
            '''
            pixel[0] = (pixel[0] - miniI) * ((maxoI-minoI) / (maxiI-miniI)) + minoI  
            pixel[1] = (pixel[1] - miniI) * ((maxoI-minoI) / (maxiI-miniI)) + minoI 
            pixel[2] = (pixel[2] - miniI) * ((maxoI-minoI) / (maxiI-miniI)) + minoI 
              
            stretched_image[i, j] = pixel 
    return stretched_image

'''
# safayat's contrast stretching 
def pixValue(pixel,x1,y1,x2,y2):
    

    if (0 <= pixel and pixel <= x1):
        return (y1 / x1)*pixel
    elif (x1 < pixel and pixel <= x2):
        return ((y2 - y1)/(x2 - x1)) * (pixel - x1) + y1
    else:
        return ((255 - y2)/(255 - x2)) * (pixel - x2) + y2
x1 = 110
y1 = 0
x2 = 175
y2 = 240    
pixValue_vec = np.vectorize(pixValue)
stretched_image = pixValue_vec(intensity_transformation_img,x1,y1,x2,y2)
plotting_function(stretched_image)
'''
#########################################################################################
def Difference_Image(image1, image2):
    # Ensure the images have the same size
    image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
    # Compute the difference image
    difference_image = cv2.absdiff(image1, image2)
    return difference_image
#########################################################################################
############################## Histogram Equalization ###################################

def Histogram_equalization(img):
    equalize = cv2.equalizeHist(img)
    return equalize

def histogram_specification(image, desired_histogram):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate the histogram of the input image
    hist, bins = np.histogram(gray.flatten(), 256, [0, 256])
    
    # Calculate the cumulative distribution function (CDF) of the input histogram
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    
    # Calculate the cumulative distribution function (CDF) of the desired histogram
    desired_cdf = desired_histogram.cumsum()
    desired_cdf_normalized = desired_cdf * hist.max() / desired_cdf.max()
    
    # Perform histogram specification
    lut = np.interp(cdf_normalized, desired_cdf_normalized, np.arange(256))
    image_equalized = lut[gray]
    
    return image_equalized

desired_hist,bins = np.histogram(hist_desire.flatten(),256,[0,256])
image_specified = histogram_specification(hist_match, desired_hist)

'''
# Load the desired histogram (e.g., from a reference image)
desired_histogram, _ = np.histogram(desired_image.flatten(), 256, [0, 256])
# Perform histogram specification
image_specified = histogram_specification(image, desired_histogram)
'''
################################# Filter #######################################

def Filter(img):
    median = cv2.medianBlur(img,5)
    bilateral = cv2.bilateralFilter(img, 20, 40, 100, borderType = cv2.BORDER_CONSTANT)
    return median,bilateral

def doggo_human_face_overlap():
    from PIL import Image
    foregnd_img = Image.open('foregnd.png').convert('RGB').resize((600,600))
    backgnd_img = Image.open('backgnd.png').convert('RGB').resize((600,600))
    masking = Image.open('alpha_chnl.png').convert('L').resize((600,600))
    Image.composite(foregnd_img, backgnd_img, masking).save('Output.png')
    out_img = cv2.imread('Output.png')
    cv2.imshow('Output Image', out_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()








    

