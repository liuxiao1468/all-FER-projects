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
save_gen = open("gen_selection.pkl","rb")
save_gen = pickle.load(save_gen)

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

# example data
x = np.arange(1, 100)
y1 = np.asarray(acc_avg)
y2 = np.asarray(num_avg)

y3 = np.asarray(acc_avg1)
y4 = np.asarray(num_avg1)
# # example error bar values that vary with x-position
# error = 0.1 + 0.2 * x
# error bar values w/ different -/+ errors
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

# fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)

plt.subplot(2, 1, 1)
plt.grid(True)
plt.errorbar(x, y1, yerr=asymmetric_error, fmt='o',color = '#196193',label='7-class')
plt.errorbar(x, y3, yerr=asymmetric_error3, fmt='*',color = '#439fdf',label='8-class')
plt.title('Average Validation Accuracy(ck+)')
plt.xlabel("Generation Number")
plt.ylabel("Validation Accuracy")
plt.ylim(0.4,1.0)
# plt.legend()
plt.legend(loc='center left', bbox_to_anchor=(0.9, 0.1))
plt.subplot(2, 1, 2)
plt.grid(True)
plt.errorbar(x, y2, yerr=asymmetric_error1, fmt='o',color = '#196193',label='7-class')
plt.errorbar(x, y4, yerr=asymmetric_error4, fmt='*',color = '#439fdf',label='8-class')
plt.title('Number of landmarks in GA')
plt.xlabel("Generation Number")
plt.ylabel("Number of Landmarks")
plt.legend(loc='center left', bbox_to_anchor=(0.9, 0.1))
plt.show()
