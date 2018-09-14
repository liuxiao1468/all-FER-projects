import cv2, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from random import *

def increase_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    # v[v > lim] = 255
    # v[v <= lim] += value

    v[v <= lim] = 255
    v[v > lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def cannyEdge(img, minT, maxT):
	# minT = 250
	# maxT = 255
	edge = cv2.Canny(img, minT, maxT)
	cv2.imshow("edge",edge)
	return edge

def img_preprocessing(img):
	bright_contrast=240
	# edge_initial=randint(80,120)
	# edge_increment=randint(55,75)
	# bright_contrast=randint(10,200)
	edge_initial=155
	edge_increment=100
	PF = np.array([bright_contrast, edge_initial, edge_increment], dtype=np.uint8)
	# show_image = img.reshape(48,48)
	# # plt.imshow(show_image, cmap='gray')
	# # plt.show()
	# plt.imsave('img.png', show_image, cmap='gray')
	# img = cv2.imread('img.png')
	cv2.imshow('original img',img)
	print ('original',img)

	img = increase_brightness(img, value=PF[0])
	# print ('after preprocessing',img)
	cv2.imshow('img',img)
	cv2.imwrite('img_pre1.png',img)
	print('brightness',img)
	edge= cannyEdge(img, minT=PF[1:2], maxT=PF[1:2]+PF[2:3])
	shape=edge.shape
	print('edges',shape)
	print('edges',edge)
	# edge = edge.reshape(-1)
	cv2.imwrite('img_pre2.png',edge)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return edge,PF

# bright_contrast=randint(20,150)
# edge_initial=randint(80,120)
# edge_increment=randint(55,75)
# PF = np.array([bright_contrast, edge_initial, edge_increment], dtype=int)
img = cv2.imread('img00.png')

[edge,PF] = img_preprocessing(img)

print(PF)


