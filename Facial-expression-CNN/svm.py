import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import svm
from matplotlib import style
import pandas as pd
from random import *
from mpl_toolkits.mplot3d import Axes3D
# # x = [1,5,1.5,8,1,9]
# # y = [2,8,1.8,8,0,11]
# # plt.scatter(x,y)
# # plt.show()

# X= np.array([[1,2],
# 	[5,8],
# 	[1.5,1.8],
# 	[8,8],
# 	[1,0],
# 	[9,11]])
# Y= [0,1,0,1,0,1]


# clf = svm.SVC(kernel = 'linear', C=1.0)
# clf.fit(X,Y)

# print(clf.predict([[0.58,0.76]]))
# print(clf.predict([[10.58,10.76]]))

# w = clf.coef_[0]
# print(w)

# a= -w[0]/w[1]
# xx = np.linspace(0,12)
# yy = a* xx -clf.intercept_[0]/w[1]

# h0 = plt.plot(xx,yy,'k-',label="non weight div")

# plt.scatter(X[:,0],X[:,1], c=Y)
# plt.legend()
# plt.show()


# PF=[]
# score=[]
# area=[]

# for m in range(5):
#     PF_save = np.loadtxt('PF_value'+str(m)+'.txt', dtype=int)
#     score_save = np.loadtxt('score_value'+str(m)+'.txt', dtype=int)
#     area_save = np.loadtxt('areas_value'+str(m)+'.txt', dtype=float)
#     PF = np.append(PF,PF_save)
#     score = np.append(score,score_save)
#     area = np.append(area,area_save)
# print(PF.shape)
# print(score.shape)
# print(area.shape)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


# Our dataset and targets
X = np.c_[(.4, -.7),
          (-1.5, -1),
          (-1.4, -.9),
          (-1.3, -1.2),
          (-1.1, -.2),
          (-1.2, -.4),
          (-.5, 1.2),
          (-1.5, 2.1),
          (1, 1),
          # --
          (1.3, .8),
          (1.2, .5),
          (.2, -2),
          (.5, -2.4),
          (.2, -2.3),
          (0, -2.7),
          (1.3, 2.1)].T
Y = [0] * 8 + [1] * 8

# figure number
fignum = 1

# fit the model
for kernel in ('linear', 'poly', 'rbf'):
    clf = svm.SVC(C=10,kernel=kernel, gamma=0.00001)
    clf.fit(X, Y)

    # plot the line, the points, and the nearest vectors to the plane
    plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    print(clf.support_vectors_[:, 0])

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                facecolors='none', zorder=10, edgecolors='k')
    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,
                edgecolors='k')

    plt.axis('tight')
    x_min = -3 
    x_max = 3
    y_min = -3
    y_max = 3

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.figure(fignum, figsize=(4, 3))
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xticks(())
    plt.yticks(())
    fignum = fignum + 1
plt.show()