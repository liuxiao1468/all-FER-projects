from Tkinter import*
import cv2
import PIL.Image, PIL.ImageTk
import time
#import real_time_class
import glob
import random
import dlib
import numpy as np
import math
import itertools
from sklearn.svm import SVC
import PIL
#from PIL import Image
from sklearn.externals import joblib
import thread
import time

class sub_in:
    def __init__(self):

        #global switch

        self.window = Toplevel()
        self.window.title('Real Time Detection')
  
 
         # open video source (by default this will try to open the computer webcam)
    
        self.vid = MyVideoCapture()
        global v
        v = StringVar()
        self.label= Label(self.window, textvariable=v)
        self.label.config(font='times 18 bold',fg='black')
        self.label.pack(pady=(10,10))
        v.set('Facial Expression')

       
 
         # Create a canvas that can fit the above video source size
        self.canvas = Canvas(self.window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack(side=TOP)



        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()


        self.btn3=Button(self.window, text="Stop", width=15, command=self.stop_detection)
        self.btn3.pack(side=LEFT,padx=(5,20),pady=(10,10))

        self.btn1=Button(self.window, text="Start", width=15, command=self.detect)
        self.btn1.pack(side=LEFT,padx=(80,80),pady=(10,10))

        self.btn2=Button(self.window, text="Close", width=15, command=self.quit)
        self.btn2.pack(side=LEFT,padx=(20,5),pady=(10,10))

        self.window.mainloop()


    def quit(self):

        self.switch = 0
        self.vid.close_frame()
        self.window.destroy()

    def stop_detection(self):

        self.switch = 0
        #print 'stop'


    def detect(self):
        self.switch = 1
        #sub_in.switch=1
        emotions = ["anger",  "disgust" ,"fear","happiness", "neutral", "sadness", "surprise"] #Emotion list


        #video_capture = cv2.VideoCapture(0) #Webcam object
        self.detector = dlib.get_frontal_face_detector() #Face detector
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.expression=[]

        det=self.detection1() 

    def detection1(self):


        thread.start_new_thread(self.detection, (frame,))
        #self.window.after(self.delay,self.detect)
        #print 'hi'
        self.window.after(300,self.detection1)


 
    def update(self):
        # Get a frame from the video source
        global frame
        ret, frame = self.vid.get_frame()
 
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = NW)

        self.window.after(self.delay, self.update)

    

    def detection(self,frame):
        #print GUI_m.
        
        w1=0.75
        w2=1-w1
        if self.switch==1:
            #print sub_in.switch
            #print 'hi'
            realtime_data = np.array([])
            #ret, frame = self.vid.read()
            
            start = time.time()
            #frame = cv2.resize(frame, (320,240))

            [xlist, ylist] = self.get_landmarks(frame)
            vec_landmark = self.get_vectorized_landmark(frame)*w1
           
            if (xlist.size) and (vec_landmark.size):
                Norm_AU_feature = self.extract_AU(xlist,ylist)*w2
                vec_AU = np.concatenate((Norm_AU_feature,vec_landmark))
                vec_AU = ((vec_AU-np.min(vec_AU))/np.ptp(vec_AU))
                realtime_data = np.concatenate((realtime_data,vec_AU))


                clf = joblib.load('best_landmark_SVM.pkl') 
                Y = clf.predict([realtime_data])
                self.label.config(fg="red")
                #print 'hi'
                if Y == 0:
                    v.set("anger")
                    self.expression.append(0)
                if Y ==1:
                    v.set("disgust")
                    self.expression.append(1)
                if Y == 2:
                    v.set("fear")
                    self.expression.append(2)
                if Y ==3:
                    v.set("happiness")
                    self.expression.append(3)
                if Y==4:
                    v.set("neutral")
                    self.expression.append(4)
                if Y ==5:
                    v.set("sadness")
                    self.expression.append(5)
                if Y==6:
                    v.set("surprise")
                    self.expression.append(6)
            end = time.time()
            elapsed = end - start
            print("time",elapsed)


            np.savetxt('Facial_expression.txt', self.expression,fmt='%.1d')
            #print self.expression


        else:
            self.label.config(fg="black")
            v.set('Facial Expression')
        #self.window.after(300,self.detection)



    def get_landmarks(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_image = clahe.apply(gray)
        detections = self.detector(clahe_image, 1)
        for k,d in enumerate(detections): #For all detected face instances individually
            shape = self.predictor(clahe_image, d) #Draw Facial Landmarks with the predictor class
            xlist = []
            ylist = []
            landmarks= []
            for i in range(0,68): #Store X and Y coordinates in two lists
                cv2.circle(clahe_image, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) 
                #For each point, draw a red circle with thickness2 on the original frame
                xlist.append(float(shape.part(i).x))
                ylist.append(float(shape.part(i).y))

            xmean = np.mean(xlist) #Find both coordinates of centre of gravity
            ymean = np.mean(ylist)
            x_max = np.max(xlist)
            x_min = np.min(xlist)
            y_max = np.max(ylist)
            y_min = np.min(ylist)
            cv2.rectangle(clahe_image,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(255,150,0),2)

            cv2.circle(clahe_image, (int(xmean), int(ymean) ), 1, (0,255,255), thickness=2)

            x_start = int(x_min-5)
            y_start = int(y_min-((ymean - y_min)/3))
            w = int(x_max+5) - x_start
            h = int(y_max+5) - y_start

            xlist[:] = [x-x_start for x in xlist]
            ylist[:] = [y-y_start for y in ylist]
 
            xlist = np.array(xlist,dtype = np.float64)
            ylist = np.array(ylist,dtype = np.float64)


        if len(detections) > 0:
            return xlist, ylist
        else: #If no faces are detected, return error message to other function to handle
            xlist = np.array([])
            ylist = np.array([])
            return xlist, ylist


    def linear_interpolation(self,xlist,ylist):
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
        return x, y


    def extract_AU(self,xlist,ylist):
        AU_feature = []
        Norm_AU_feature = []
        AU1_1_x = xlist[19:22]
        AU1_1_y = ylist[19:22]
        AU1_1_x,AU1_1_y = self.linear_interpolation(AU1_1_x,AU1_1_y)
        AU1_1_x,AU1_1_y = self.linear_interpolation(AU1_1_x,AU1_1_y)
        AU_feature = self.get_average_curvature(AU1_1_x,AU1_1_y)

        AU1_2_x = xlist[22:25]
        AU1_2_y = ylist[22:25]
        AU1_2_x,AU1_2_y = self.linear_interpolation(AU1_2_x,AU1_2_y)
        AU1_2_x,AU1_2_y = self.linear_interpolation(AU1_2_x,AU1_2_y)
        AU_feature = AU_feature + self.get_average_curvature(AU1_2_x,AU1_2_y)

        AU2_1_x = xlist[17:20]
        AU2_1_y = ylist[17:20]
        AU2_1_x,AU2_1_y = self.linear_interpolation(AU2_1_x,AU2_1_y)
        AU2_1_x,AU2_1_y = self.linear_interpolation(AU2_1_x,AU2_1_y)
        AU_feature = AU_feature + self.get_average_curvature(AU2_1_x,AU2_1_y)
        AU2_2_x = xlist[24:27]
        AU2_2_y = ylist[24:27]
        AU2_2_x,AU2_2_y = self.linear_interpolation(AU2_2_x,AU2_2_y)
        AU2_2_x,AU2_2_y = self.linear_interpolation(AU2_2_x,AU2_2_y)
        AU_feature = AU_feature + self.get_average_curvature(AU2_2_x,AU2_2_y)

        AU5_1_x = xlist[36:40]
        AU5_1_y = ylist[36:40]
        AU5_1_x,AU5_1_y = self.linear_interpolation(AU5_1_x,AU5_1_y)
        AU5_1_x,AU5_1_y = self.linear_interpolation(AU5_1_x,AU5_1_y)
        AU_feature = AU_feature + self.get_average_curvature(AU5_1_x,AU5_1_y)
        AU5_2_x = xlist[42:46]
        AU5_2_y = ylist[42:46]
        AU5_2_x,AU5_2_y = self.linear_interpolation(AU5_2_x,AU5_2_y)
        AU5_2_x,AU5_2_y = self.linear_interpolation(AU5_2_x,AU5_2_y)
        AU_feature = AU_feature + self.get_average_curvature(AU5_2_x,AU5_2_y)

        AU7_1_x = np.append(xlist[39:42],xlist[36])
        AU7_1_y = np.append(ylist[39:42],ylist[36])
        AU7_1_x,AU7_1_y = self.linear_interpolation(AU7_1_x,AU7_1_y)
        AU7_1_x,AU7_1_y = self.linear_interpolation(AU7_1_x,AU7_1_y)
        AU_feature = AU_feature + self.get_average_curvature(AU7_1_x,AU7_1_y)

        AU7_2_x = np.append(xlist[46:48],xlist[42])
        AU7_2_y = np.append(ylist[46:48],ylist[42])
        AU7_2_x,AU7_2_y = self.linear_interpolation(AU7_2_x,AU7_2_y)
        AU7_2_x,AU7_2_y = self.linear_interpolation(AU7_2_x,AU7_2_y)
        AU_feature = AU_feature + self.get_average_curvature(AU7_2_x,AU7_2_y)

        AU9_x = xlist[31:36]
        AU9_y = ylist[31:36]
        AU9_x,AU9_y = self.linear_interpolation(AU9_x,AU9_y)
        AU9_x,AU9_y = self.linear_interpolation(AU9_x,AU9_y)
        AU_feature = AU_feature + self.get_average_curvature(AU9_x,AU9_y)

        AU10_x = np.append(xlist[48:51],xlist[52:55])
        AU10_y = np.append(ylist[48:51],ylist[52:55])
        AU10_x,AU10_y = self.linear_interpolation(AU10_x,AU10_y)
        AU10_x,AU10_y = self.linear_interpolation(AU10_x,AU10_y)
        AU_feature = AU_feature + self.get_average_curvature(AU10_x,AU10_y)

        AU12_1_x = [xlist[48]] + [xlist[60]] + [xlist[67]]
        AU12_1_y = [ylist[48]] + [ylist[60]] + [ylist[67]]
        AU12_1_x,AU12_1_y = self.linear_interpolation(AU12_1_x,AU12_1_y)
        AU12_1_x,AU12_1_y = self.linear_interpolation(AU12_1_x,AU12_1_y)
        AU_feature = AU_feature + self.get_average_curvature(AU12_1_x,AU12_1_y)

        AU12_2_x = [xlist[54]] + [xlist[64]] + [xlist[65]]
        AU12_2_y = [ylist[54]] + [ylist[64]] + [ylist[65]]
        AU12_2_x,AU12_2_y = self.linear_interpolation(AU12_2_x,AU12_2_y)
        AU12_2_x,AU12_2_y = self.linear_interpolation(AU12_2_x,AU12_2_y)
        AU_feature = AU_feature + self.get_average_curvature(AU12_2_x,AU12_2_y)


        AU20_x = xlist[55:60]
        AU20_y = ylist[55:60]
        AU20_x,AU20_y = self.linear_interpolation(AU20_x,AU20_y)
        AU20_x,AU20_y = self.linear_interpolation(AU20_x,AU20_y)
        AU_feature = AU_feature + self.get_average_curvature(AU20_x,AU20_y)

        Norm_AU_feature = (AU_feature-np.min(AU_feature))/np.ptp(AU_feature)


        return Norm_AU_feature


    def get_average_curvature(self,AU_xlist,AU_ylist):
        K = []
        Z = np.polyfit(AU_xlist,AU_ylist,4)
        P = np.poly1d(Z)
        P_1 = np.poly1d.deriv(P)
        P_2 = np.poly1d.deriv(P_1)
        for i in range(len(AU_xlist)):

            Y = 1+math.pow(P_1(AU_xlist[i]),2)
            Y = math.pow(Y,1.5)
            K.append(P_2(AU_xlist[i])/Y)
        # m_K = np.mean(K)
        m_K = K
        return m_K

        

    def get_vectorized_landmark(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_image = clahe.apply(gray)
        detections = self.detector(clahe_image, 1)
        for k,d in enumerate(detections): #For all detected face instances individually
            shape = self.predictor(image, d) #Draw Facial Landmarks with the predictor class
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

            landmarks_dist = landmarks_dist[17:27]+landmarks_dist[31:40]+ landmarks_dist[42:51]+ [landmarks_dist[60]]+[landmarks_dist[67]]+ [landmarks_dist[64]]+ [landmarks_dist[65]]+ landmarks_dist[52:60]
            landmarks_theta = landmarks_theta[17:27]+landmarks_theta[31:40]+ landmarks_theta[42:51]+ [landmarks_theta[60]]+[landmarks_theta[67]]+ [landmarks_theta[64]]+ [landmarks_theta[65]]+ landmarks_theta[52:60]
            landmarks_dist = np.array(landmarks_dist,dtype = np.float64)
            Norm_landmarks_dist = (landmarks_dist-np.min(landmarks_dist))/np.ptp(landmarks_dist)
            landmarks_theta = np.array(landmarks_theta,dtype = np.float64)
            Norm_landmarks_theta = (landmarks_theta-np.min(landmarks_theta))/np.ptp(landmarks_theta)

            landmarks_vectorized =  np.concatenate((Norm_landmarks_dist,Norm_landmarks_theta))
            return landmarks_vectorized
        if len(detections) < 1:
            landmarks_vectorized = np.array([])
        return landmarks_vectorized

    


class MyVideoCapture:
    def __init__(self):
        # Open the video source
        #video_index='/home/liutao/Downloads/DanielNew2.mp4'
        self.vid = cv2.VideoCapture(0)
        #self.vid = cv2.VideoCapture(video_index)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        #global ret, frame
        if self.vid.isOpened():

            ret, frame = self.vid.read()
            if ret:
                            # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    def close_frame(self):
        if self.vid:
            self.vid.release()
