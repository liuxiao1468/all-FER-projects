import cv2
import glob
import random
import math
import numpy as np
import dlib
import itertools
from sklearn.svm import SVC

# a = np.array([])
# b = np.array([])
# c = np.array([1.222,2.111,3.2,4])
# for i in range (3):
#     if a.size:
#         a = np.vstack((a,c))
#     else:
#         a = np.concatenate((a,c))

# print(a)
# # for i in range (3):
# #     b = np.concatenate((b,[1.4]))
# # print(b)
# np.savetxt('a.txt', a, fmt='%1.4e')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
image = cv2.imread('surprise.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_image = clahe.apply(gray)
detections = detector(clahe_image, 1)

for k,d in enumerate(detections): #For all detected face instances individually
    shape = predictor(clahe_image, d) #Draw Facial Landmarks with the predictor class
    xlist = []
    ylist = []
    landmarks= []
    for i in range(0,68): #Store X and Y coordinates in two lists
        cv2.circle(image, (shape.part(i).x, shape.part(i).y), 1, (0,255,255), thickness=2) 
        #For each point, draw a red circle with thickness2 on the original frame
        xlist.append(float(shape.part(i).x))
        ylist.append(float(shape.part(i).y))

    xmean = np.mean(xlist) #Find both coordinates of centre of gravity
    ymean = np.mean(ylist)
    x_max = np.max(xlist)
    x_min = np.min(xlist)
    y_max = np.max(ylist)
    y_min = np.min(ylist)
    cv2.rectangle(image,(int(x_min),int(y_min-((ymean - y_min)/3))),(int(x_max),int(y_max)),(211,211,211),2)
    print ("centre of gravity",xmean, ymean)
    print ("range of the face",x_max, x_min, y_max, y_min)
    cv2.circle(image, (int(xmean), int(ymean) ), 1, (0,0,255), thickness=2) 

    x_start = int(x_min)
    y_start = int(y_min-((ymean - y_min)/3))
    w = int(x_max) - x_start
    h = int(y_max) - y_start

    crop_img = image[y_start:y_start+h, x_start:x_start+w] # Crop from {x, y, w, h } => {0, 0, 300, 400}

    # cv2.imshow("cropped", crop_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imshow("image", image) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('crop_example.png',crop_img)
    cv2.imwrite('landmark_example.png',image)

# import PIL
# from PIL import Image

# mywidth = 255
# hsize = 255

# img = Image.open('cropped.png')
# # wpercent = (mywidth/float(img.size[0]))
# # hsize = int((float(img.size[1])*float(wpercent)))
# img = img.resize((mywidth,hsize), PIL.Image.ANTIALIAS)
# img.save('resized.jpg')
# print(img.size)
    # xlist = np.array(xlist,dtype = np.float64)
    # ylist = np.array(ylist,dtype = np.float64)
    # xlist = np.float32(xlist)
    # ylist = np.float32(ylist)

