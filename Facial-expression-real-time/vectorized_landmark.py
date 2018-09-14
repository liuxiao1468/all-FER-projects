
import cv2
import glob
import random
import math
import numpy as np
import dlib
import itertools
from sklearn.svm import SVC


def get_vectorized_landmark(image):
    detections = detector(image, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(0,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        xcentral = [(x-xmean) for x in xlist]
        ycentral = [(y-ymean) for y in ylist]
        landmarks_dist = []
        landmarks_theta = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            # landmarks_vectorized.append(w)
            # landmarks_vectorized.append(z)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp)

            landmarks_dist.append(dist)
            landmarks_theta.append((math.atan2(y, x)*360)/(2*math.pi))

        landmarks_dist = landmarks_dist[17:]
        landmarks_theta = landmarks_theta[17:]
        landmarks_dist = np.array(landmarks_dist,dtype = np.float64)
        Norm_landmarks_dist = (landmarks_dist-np.min(landmarks_dist))/np.ptp(landmarks_dist)
        landmarks_theta = np.array(landmarks_theta,dtype = np.float64)
        Norm_landmarks_theta = (landmarks_theta-np.min(landmarks_theta))/np.ptp(landmarks_theta)

        landmarks_vectorized =  np.concatenate((Norm_landmarks_dist,Norm_landmarks_theta))
        return landmarks_vectorized
    if len(detections) < 1:
        landmarks_vectorized = np.array([])
    return landmarks_vectorized


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Or set this to whatever you named the downloaded file
# # clf = SVC(kernel='linear', probability=True, tol=1e-3)
# #, verbose = True) #Set the classifier as a support vector machines with polynomial kernel

image = cv2.imread('image.png')
landmarks_vectorized = get_vectorized_landmark(image)

emotions = ["anger",   "happiness", "neutral", "sadness", "surprise"]
for emotion in emotions:
    print(emotions.index(emotion))


print(landmarks_vectorized)


