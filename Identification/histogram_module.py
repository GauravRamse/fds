import numpy as np
from numpy import histogram as hist
import math
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
filteringpath = os.path.join(parentdir, 'Filtering')
sys.path.insert(0,filteringpath)
import gauss_module

#  compute histogram of image intensities, histogram should be normalized so that sum of all values equals 1
#  assume that image intensity varies between 0 and 255
#
#  img_gray - input image in grayscale format
#  num_bins - number of bins in the histogram
def normalized_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'
    arr_img=np.array(img_gray.flatten())
    #dictionary to be used for hist with keys=num_bins and values 0
    hist=dict((name,0) for name in range(num_bins)) 
    #filling in the dictionary by adding each intensity in the array in its bin interval
    for i in arr_img: #iteration in flatten array
        index=int(math.floor(i/(256/num_bins))) #the index value reprents which bin includes that intensity value
        hist[index]+=1 #sum value is associated to a key value (bin index).
    #now we normalize each value in the dictionary (bin) for the total sum of pixels
    norm_hist=[]
    for value in hist.values(): 
        norm_hist.append(value/len(arr_img))
    #now we return the output in the format need fot plotting (list of values)
    bins = np.array(range(num_bins))
    hist = np.array(norm_hist)
    return hist, bins

#  Compute the *joint* histogram for each color channel in the image
#  The histogram should be normalized so that sum of all values equals 1
#  Assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^3
#
#  E.g. hists[0,9,5] contains the number of image_color pixels such that:
#       - their R values fall in bin 0
#       - their G values fall in bin 9
#       - their B values fall in bin 5
def rgb_hist(img_color_double, num_bins):
    assert len(img_color_double.shape) == 3, 'image dimension mismatch'
    assert img_color_double.dtype == 'float', 'incorrect image type'    #Define a 3D histogram  with "num_bins^3" number of entries
    #initializing dictionaries with keys (bin indexes)
    r_hist=dict((name,0) for name in range(num_bins))
    g_hist=dict((name,0) for name in range(num_bins)) 
    b_hist=dict((name,0) for name in range(num_bins))
    #for each pixel in the input image we extraxt the three values of r, g and b
    # Loop for each pixel i in the image
    for i in range(img_color_double.shape[0]*img_color_double.shape[1]): #iteration in len(X)*len(Y) of the image matrix
        row=int(i//img_color_double.shape[0]) #index of row is the i index divided by the number of pixels in each row, so it increases by one each len(X) number of pixels
        col=int(i-row*img_color_double.shape[0]) #index of col is the value of i minus the number of pxels in the rows which have already been iterated
        # Increment the histogram bin which corresponds to the R,G,B value of the pixel i
        r,g,b= img_color_double[row,col] #unpack each value for r,g and b
        # 255/num_bins is the number of color itervals represented in each bin. If we divide the value for a specific color in a specific pixel by this value and take the lower bound we have the index of the bin that must be incremented by 1
        r_key=int(math.floor(r/(256/num_bins)))
        g_key=int(math.floor(g/(256/num_bins)))
        b_key=int(math.floor(b/(256/num_bins)))
        r_hist[r_key]+=1
        g_hist[g_key]+=1
        b_hist[b_key]+=1
    hists_list=[]
    #we concatenate the three color channels
    for hist in [r_hist,g_hist,b_hist]: 
        hists_list += hist.values()  
    #Normalize the histogram such that its integral (sum) is equal 1
    norm_hists_list= [x/(img_color_double.shape[0]*img_color_double.shape[1]*3) for x in hists_list] #we divide by tot number of pixels*3 
    #Return the histogram as a 1D vector
    hists=np.array(norm_hists_list)
    return hists


#  Compute the *joint* histogram for the R and G color channels in the image
#  The histogram should be normalized so that sum of all values equals 1
#  Assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^2
#
#  E.g. hists[0,9] contains the number of image_color pixels such that:
#       - their R values fall in bin 0
#       - their G values fall in bin 9
def rg_hist(img_color_double, num_bins):
    assert len(img_color_double.shape) == 3, 'image dimension mismatch'
    assert img_color_double.dtype == 'float', 'incorrect image type'    #Define a 3D histogram  with "num_bins^3" number of entries
    #same procedure of rgb_hist    
    r_hist=dict((name,0) for name in range(num_bins)) 
    g_hist=dict((name,0) for name in range(num_bins)) 
   
    for i in range(img_color_double.shape[0]*img_color_double.shape[1])
        row=int(i//img_color_double.shape[0])
        col=int(i-row*img_color_double.shape[0]) 
        r,g,b= img_color_double[row,col] 
        r_key=int(math.floor(r/(256/num_bins)))
        g_key=int(math.floor(g/(256/num_bins)))
        r_hist[r_key]+=1
        g_hist[g_key]+=1   
    
    hists_list=[]
    for hist in [r_hist,g_hist]: 
        hists_list += hist.values()    
    
    norm_hists_list= [x/(img_color_double.shape[0]*img_color_double.shape[1]*2) for x in hists_list]

    hists=np.array(norm_hists_list)
    return hists

#  Compute the *joint* histogram of Gaussian partial derivatives of the image in x and y direction
#  Set sigma to 3.0 and cap the range of derivative values is in the range [-6, 6]
#  The histogram should be normalized so that sum of all values equals 1
#
#  img_gray - input gray value image
#  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
#
#  Note: you may use the function gaussderiv from the Filtering exercise (gauss_module.py)
def dxdy_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'
    sigma=3.0
    
    #call gaussian derivative to obtain the two grey images
    img_gray_dx, img_gray_dy = gauss_module.gaussderiv(img_gray,sigma)
    #we want to consider values out of the range[-6:6] as the extremes (-6 and 6)
    #In order to compute the normalized histogram we need values in the 0-255 range, so we scale the original values (range[-6:6]) in order to obtain values that are scaled in the input format of normalized_hist (range[0:255])
    for img in [img_gray_dx,img_gray_dy]: #we do the same on images which have been dirived in both direction x and y
      for i in range(len(img)):
        for j in range(len(img[i])):
          e=img[i][j]
          #we check that values are in the [-6:6] range
          if e>6:
            img[i][j]=6
          elif e<-6:
            img[i][j]=-6
          #we scale values by multiplying them for the scaling factor 255/6
          img[i][j]=img[i][j]*255/6
            
    #Compute a normalized histogram for each of the two grey images
    #We use absolute values since for edge detection we are not interested in the fact that we move to darker areas to light areas or the contrary
    hist_gray_dx = normalized_hist(np.absolute(img_gray_dx),num_bins)[0] #[0] since normalized_hist returns two values
    hist_gray_dy = normalized_hist(np.absolute(img_gray_dy),num_bins)[0]
    # we concatenate two flattened arrays
    hists=list(hist_gray_dx.flatten())+list(hist_gray_dy.flatten())
    
    #Return the histogram as a 1D vector
    hists=np.array(hists)/2 #normalization of the dx and dy normalized histograms by dividing by two
    hists = hists.reshape(hists.size)
    return hists

def is_grayvalue_hist(hist_name):
  if hist_name == 'grayvalue' or hist_name == 'dxdy':
    return True
  elif hist_name == 'rgb' or hist_name == 'rg':
    return False
  else:
    assert False, 'unknown histogram type'


def get_hist_by_name(img, num_bins_gray, hist_name):
  if hist_name == 'grayvalue':
    return normalized_hist(img, num_bins_gray)
  elif hist_name == 'rgb':
    return rgb_hist(img, num_bins_gray)
  elif hist_name == 'rg':
    return rg_hist(img, num_bins_gray) #img is type sting, should be array
  elif hist_name == 'dxdy':
    return dxdy_hist(img, num_bins_gray)
  else:
    assert False, 'unknown distance: %s'%hist_name
