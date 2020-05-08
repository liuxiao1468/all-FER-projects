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

index = save_score.index(max(save_score))
fitness_value = save_score[index]
print(fitness_value)
print(save_gen[index])
print("the best num of landmarks: ",sum(save_gen[index]))
for i in range (len(save_gen)):
    print(save_score[i])
