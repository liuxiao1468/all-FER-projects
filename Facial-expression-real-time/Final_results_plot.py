

# import PIL
# from PIL import Image

# mywidth = 255
# hsize = 100

# img = Image.open('cropped.png')
# # wpercent = (mywidth/float(img.size[0]))
# # hsize = int((float(img.size[1])*float(wpercent)))
# img = img.resize((mywidth,hsize), PIL.Image.ANTIALIAS)
# img.save('resized.jpg')


import cv2
import glob
import random
import dlib
import numpy as np
import math
import itertools
from sklearn.svm import SVC
import PIL
from PIL import Image
from sklearn.externals import joblib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


CC = [-1,0,1,1.69,2,2.69,3, 3.48, 4,4.47]
accuracy = np.array([])
x_axis = np.array([])
y_axis = np.array([])
for i in range (10):
    C = [0.1,1,10,50,100,500,1000,3000,10000,30000]
    result = np.loadtxt('/home/leo/woody_vision/modified/'+'C'+str(C[i])+'.txt', dtype = np.float64)
    # result = np.loadtxt('/home/leo/woody_vision/final_results/'+'C'+str(C[i])+'.txt', dtype = np.float64)
    
    accuracy = np.concatenate((accuracy,result))
print(accuracy.shape)

for j in range (10):
	w = []
	for i in range (18):
		w1 = 0.05*(i+1)
		w = w+[w1]
	x_axis = np.concatenate((x_axis,w))
print(x_axis.shape)

for i in range (10):
	c = []
	for j in range (18):
		c = c+[CC[i]]
	y_axis = np.concatenate((y_axis,c))
print(y_axis.shape) 

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_trisurf(x_axis, y_axis, accuracy, linewidth=0.2, antialiased=True,cmap='viridis', edgecolor='none')
# ax.set_title('Accuracy with varied Weights and C value')
ax.set_xlabel('W1')
ax.set_ylabel('C Value (10^y)')
ax.set_zlabel('Accuracy')
# fig.colorbar(surf, shrink=0.4, aspect=5)
fig.savefig('demo.png', transparent=True)

plt.show()






