import numpy as np
import math



# Compute the intersection distance between histograms x and y
# Return 1 - hist_intersection, so smaller values correspond to more similar histograms
# Check that the distance range in [0,1]

def dist_intersect(x,y):
  sum_q=0
  sum_v=0
  sum_min=0
   #iterate both arrays and compute the summation of the min vlaue between each couple divided by the sum of the values of each istogram and divided by 2
  for j in range(len(x)): #they have same len 
    q=x[j]
    v=y[j]
    sum_q+=q
    sum_v+=v
    sum_min+=min([q,v])
    #sum_q and sum_v are in the denominator so they must be != 0, in case they are zero we assign to them a very small value not to lose any information

  if sum_q==0:
    first_part= 0
  else: 
    first_part = sum_min/sum_q
  
  if sum_v==0:
    second_part = 0
  else:
    second_part= sum_min/sum_v
  dist_int= (first_part+second_part)*1/2 #applying formula for intersection
  return dist_int


# Compute the L2 distance between x and y histograms
# Check that the distance range in [0,sqrt(2)]

def dist_l2(x,y):
  sum_sq=0
  for j in range(len(x)): #they have same len 
    q=x[j]
    v=y[j] 
    sum_sq+= (q-v)**2 #applying formula for euclidean dist
  return sum_sq

# Compute chi2 distance between x and y
# Check that the distance range in [0,Inf]
# Add a minimum score to each cell of the histograms (e.g. 1) to avoid division by 0

def dist_chi2(x,y):
  sum_chi=0
  for j in range(len(x)): #they have same len 
    q=x[j]
    v=y[j]
    den=q+v
    if q==0 and v==0:
      sum_chi =0
    else:
      sum_chi+= ((q-v)**2)/(den) #denominator
    
  return sum_chi

def get_dist_by_name(x, y, dist_name):
  if dist_name == 'chi2':
    return dist_chi2(x,y)
  elif dist_name == 'intersect':
    return dist_intersect(x,y)
  elif dist_name == 'l2':
    return dist_l2(x,y)
  else:
    assert False, 'unknown distance: %s'%dist_name
