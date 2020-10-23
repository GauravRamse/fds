# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2



"""
Gaussian function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian values Gx computed at the indexes x
"""

def gauss(sigma): #sigma = 4.0
    #...
    #Generate a vector x of values on which the Gaussian filter is defined: integer values on the interval [-3*sigma, 3*sigma]
    low = -3*sigma
    high = (3*sigma)+1
    range_x = [int(i) for i in range(low,high)]
    Gx = []
    for x in range_x:
        G = (math.exp((-x ** 2) / (2 * (sigma ** 2))))* (1 / math.sqrt(2 * math.pi * sigma))
        Gx.append(G)
    return Gx, range_x

"""
Implement a 2D Gaussian filter, leveraging the previous gauss.
Implement the filter from scratch or leverage the convolve2D method (scipy.signal)
Leverage the separability of Gaussian filtering
Input: image, sigma (standard deviation)
Output: smoothed image
"""
def gaussianfilter(img, sigma):
    #...
    #col = img.size[0]
    #row = img.size[1]
    #tmp_img = np.zeros((row,col)) #define an empty matrix (img) with the same dimension of the input img
    kernel = ((13,13))
    #fill the kernel with the values on which the G filter is defined (range [-3*sigma,3*sigma])
    low = -3*sigma
    high = (3*sigma)+1
    kernel = [int(i,j) for i,j in range(low,high)]
    #extact the first col of the kernel
    Gx = kernel[:, 0]
    #extract the first row of the kernel
    Gy = kernel[0, :]    #computing the first convolution
    tmp_img = scipy.signal.conv2d(img.flatten(),Gx,mode='full', boundary='fill', fillvalue=0)
    #using img.flatten() because the first argument of the conv2d() must be an array, as the second one
    #now tmp_img (the output of the first convolution) is an array
    #computing the second convolution (on the output of the first one)
    smooth_img = scipy.signal.conv2d(tmp_img,Gy,mode='full', boundary='fill', fillvalue=0)
    #SOURCE: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html
    return smooth_img

def gaussdx(sigma):

    #...
    
    return Dx, x



def gaussderiv(img, sigma):

    #...
    
    return imgDx, imgDy

