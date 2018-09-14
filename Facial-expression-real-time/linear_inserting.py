import cv2
import glob
import random
import math
import numpy as np
import dlib
import itertools
from sklearn.svm import SVC

x = np.array([1,2,3,4,5,6])
y = np.array([7,6,5,4,3,2])
times = 2

def linear_interpolation(xlist,ylist):
    xlist = np.array(xlist,dtype = np.float64)
    ylist = np.array(ylist,dtype = np.float64)
    x_new = np.array([])
    y_new = np.array([])
    x = np.array([])
    y = np.array([])
    for i in range (len(xlist)-1):
    	x_new = np.concatenate((x_new,[(xlist[i]+xlist[i+1])/2.0]))
    	y_new = np.concatenate((y_new,[(ylist[i]+ylist[i+1])/2.0]))

    for j in range (len(xlist)):
        if j<(len(xlist)-1):
            x = np.concatenate((x,[xlist[j]]))
            x = np.concatenate((x,[x_new[j]]))
            y = np.concatenate((y,[ylist[j]]))
            y = np.concatenate((y,[y_new[j]]))
        else:
            x = np.concatenate((x,[xlist[j]]))
            y = np.concatenate((y,[ylist[j]]))

    # x = np.append(xlist, x_new)
    # ylist = np.append(ylist, y_new)

    return x, y


x,y = linear_interpolation(x,y)
x,y = linear_interpolation(x,y)
print (x)
print (y)
