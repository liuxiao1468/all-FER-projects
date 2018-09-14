import cv2
import glob
import random
import math
import numpy as np
import dlib
import itertools
from sklearn.svm import SVC

x = np.array([1,2,3,4,6,7])
y = np.array([3,2,1,2,3,1])
times = 2

def linear_interpolation(xlist,ylist):
    xlist = np.array(xlist,dtype = np.float64)
    ylist = np.array(ylist,dtype = np.float64)
    x_new = np.array([])
    y_new = np.array([])
    for i in range (len(xlist)-1):
    	x_new = np.concatenate((x_new,[(xlist[i]+xlist[i+1])/2.0]))
    	y_new = np.concatenate((y_new,[(ylist[i]+ylist[i+1])/2.0]))
    xlist = np.append(xlist, x_new)
    ylist = np.append(ylist, y_new)
    return xlist, ylist

x,y = linear_interpolation(x,y)
print (x)
print (y)
