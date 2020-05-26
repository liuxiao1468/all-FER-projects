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
import matplotlib.pyplot as plt
import joblib
import time
from imutils import face_utils

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import time
import pickle

## if you need to read
# save_gen = open("gen_selection.pkl","rb")
# save_gen = pickle.load(save_gen)

save_score = open("gen_score.pkl","rb")
save_score = pickle.load(save_score)

acc_avg = open('acc_avg.pkl','rb')
acc_avg = pickle.load(acc_avg)
acc_up = open('acc_up.pkl','rb')
acc_up = pickle.load(acc_up)
acc_low = open('acc_low.pkl','rb')
acc_low = pickle.load(acc_low)

num_avg = open('num_avg.pkl','rb')
num_avg = pickle.load(num_avg)
num_up = open('num_up.pkl','rb')
num_up = pickle.load(num_up)
num_low = open('num_low.pkl','rb')
num_low = pickle.load(num_low)


acc_avg1 = open('acc_avg1.pkl','rb')
acc_avg1 = pickle.load(acc_avg1)
acc_up1 = open('acc_up1.pkl','rb')
acc_up1 = pickle.load(acc_up1)
acc_low1 = open('acc_low1.pkl','rb')
acc_low1 = pickle.load(acc_low1)

num_avg1 = open('num_avg1.pkl','rb')
num_avg1 = pickle.load(num_avg1)
num_up1 = open('num_up1.pkl','rb')
num_up1 = pickle.load(num_up1)
num_low1 = open('num_low1.pkl','rb')
num_low1 = pickle.load(num_low1)


acc_avg3 = open('acc_avg3.pkl','rb')
acc_avg3 = pickle.load(acc_avg3)
acc_up3 = open('acc_up3.pkl','rb')
acc_up3 = pickle.load(acc_up3)
acc_low3 = open('acc_low3.pkl','rb')
acc_low3 = pickle.load(acc_low3)

num_avg3 = open('num_avg3.pkl','rb')
num_avg3 = pickle.load(num_avg3)
num_up3 = open('num_up3.pkl','rb')
num_up3 = pickle.load(num_up3)
num_low3 = open('num_low3.pkl','rb')
num_low3 = pickle.load(num_low3)

# example data
x = np.arange(1, 300)
# x1 = np.arange(1.1, 100.1)
y1 = np.asarray(acc_avg)
y2 = np.asarray(num_avg)

y3 = np.asarray(acc_avg1)
y4 = np.asarray(num_avg1)

y5 = np.asarray(acc_avg3)
y6 = np.asarray(num_avg3)


lower_error = np.asarray(acc_low)
upper_error = np.asarray(acc_up)
asymmetric_error = [lower_error, upper_error]

lower_error1 = np.asarray(num_low)
upper_error1 = np.asarray(num_up)
asymmetric_error1 = [lower_error1, upper_error1]

lower_error3 = np.asarray(acc_low1)
upper_error3 = np.asarray(acc_up1)
asymmetric_error3 = [lower_error3, upper_error3]

lower_error4 = np.asarray(num_low1)
upper_error4 = np.asarray(num_up1)
asymmetric_error4 = [lower_error4, upper_error4]


lower_error5 = np.asarray(acc_low3)
upper_error5 = np.asarray(acc_up3)
asymmetric_error5 = [lower_error5, upper_error5]

lower_error6 = np.asarray(num_low3)
upper_error6 = np.asarray(num_up3)
asymmetric_error6 = [lower_error6, upper_error6]

#3a9b85  #ffb9ff

plt.figure(1)
plt.grid(True)
plt.errorbar(x, y1, yerr=asymmetric_error, color = '#ffff88',zorder=1)
plt.plot(x, y1,'-',color = '#afaf76',label='7-class in CK+',zorder=3)
plt.errorbar(x, y3, yerr=asymmetric_error3, color = '#9effe9',zorder=1)
plt.plot(x, y3,'-',color = '#3a9b85',label='8-class in CK+',zorder=3)
plt.errorbar(x, y5, yerr=asymmetric_error5, color = '#ffe4ff',zorder=1)
plt.plot(x, y5,'-',color = '#eba5eb',label='7-class in MUG',zorder=3)


plt.title('Average Validation Accuracy')
# plt.xlabel("Generation Number")
plt.ylabel("Validation Accuracy")
plt.ylim(0.7,1.0)
# plt.legend()
plt.legend(loc='center left', bbox_to_anchor=(0.9, 0.1))

plt.figure(2)
plt.grid(True)
plt.errorbar(x, y2, yerr=asymmetric_error1,color = '#ffff88',zorder=1)
plt.plot(x, y2,'-',color = '#afaf76',label='7-class in CK+',zorder=3)
plt.errorbar(x, y4, yerr=asymmetric_error4, color = '#9effe9',zorder=1)
plt.plot(x, y4,'-',color = '#3a9b85',label='8-class in CK+',zorder=3)
plt.errorbar(x, y6, yerr=asymmetric_error6, color = '#ffe4ff',zorder=1)
plt.plot(x, y6,'-',color = '#eba5eb',label='7-class in MUG',zorder=3)
# plt.errorbar(x, y6, yerr=asymmetric_error6, fmt='^',color = '#13711b',label='7-class in MUG')
plt.title('Number of landmarks in GA')
plt.xlabel("i-th Generation")
plt.ylabel("Number of Landmarks")
plt.ylim(15,60)
plt.legend(loc='center left', bbox_to_anchor=(0.9, 0.1))
plt.show()
