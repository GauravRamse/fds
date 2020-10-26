import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import histogram_module
import dist_module

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


# model_images - list of file names of model images
# query_images - list of file names of query images
#
# dist_type - string which specifies distance type:  'chi2', 'l2', 'intersect'
# hist_type - string which specifies histogram type:  'grayvalue', 'dxdy', 'rgb', 'rg'
#
# note: use functions 'get_dist_by_name', 'get_hist_by_name' and 'is_grayvalue_hist' to obtain 
#       handles to distance and histogram functions, and to find out whether histogram function 
#       expects grayvalue or color image

def find_best_match(model_images, query_images, dist_type, hist_type, num_bins):

    #this value indicates if the rgb2gray() must be applied to the color array or not
    hist_isgray = histogram_module.is_grayvalue_hist(hist_type)


    model_hists = compute_histograms(model_images, hist_type, hist_isgray, num_bins)
    query_hists = compute_histograms(query_images, hist_type, hist_isgray, num_bins)
    
    D = np.zeros((len(model_images), len(query_images)))
    for query_index in range(len(query_hists)):
        for model_index in range(len(model_hists)):
            D[query_index][model_index]=dist_module.get_dist_by_name(query_hists[query_index],model_hists[model_index],dist_type)
    best_match =list(np.unravel_index(np.argmax(D, axis=None), D.shape))
    
    return best_match, D


def compute_histograms(image_list, hist_type, hist_isgray, num_bins):
    
    image_hist = []

    # Compute histogram for each image and add it at the bottom of image_hist
    for image in image_list:

        #Converting the image in the image_list in array
        img_color = np.array(Image.open(image))
        img_color = img_color.astype('double')
        if hist_isgray == True:
            img_gray = rgb2gray(img_color)

        #According to image_type, we compute the histogram
        if hist_type == 'grayvalue':
            image_hist.append(histogram_module.get_hist_by_name(img_gray,num_bins,hist_type))
        elif hist_type == 'rgb':
            image_hist.append(histogram_module.get_hist_by_name(img_color,num_bins,hist_type))
        elif hist_type == 'rg':
            image_hist.append(histogram_module.get_hist_by_name(img_color,num_bins,hist_type))
        elif hist_type == 'dxdy':
            image_hist.append(histogram_module.get_hist_by_name(img_gray,num_bins,hist_type))
        else:
            print ("Histogram type not valid")

    return image_hist



# For each image file from 'query_images' find and visualize the 5 nearest images from 'model_image'.
#
# Note: use the previously implemented function 'find_best_match'
# Note: use subplot command to show all the images in the same Python figure, one row per query image

def show_neighbors(model_images, query_images, dist_type, hist_type, num_bins):
    
    
    plt.figure()

    num_nearest = 5  # show the top-5 neighbors
    
    #... (your code here)

