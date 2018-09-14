import glob
from shutil import copyfile
emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happiness", "sadness", "surprise"] #Define emotion order
participants = glob.glob("/home/liutao/landmark-FER-SVM/CK+/Emotion//*") #Returns a list of all folders with participant numbers
for x in participants:
    part = "%s" %x[-4:] #store current participant number
    #print(part)
    for sessions in glob.glob("%s//*" %x): #Store list of sessions for current participant
        for files in glob.glob("%s//*" %sessions): #emotion text (index number)
            # current_session = files[20:-30]
            current_session = sessions[-3:]
            # print(current_session)
            file = open(files, 'r')
            emotion = int(float(file.readline())) #emotions are encoded as a float, readline as float, then convert to integer.
            #print emotion
            # for i in range (1):
            # for name in sorted(glob.glob("/home/leo/deeplearning/CK+/extended-cohn-kanade-images/cohn-kanade-images//%s//%s//*" %(part, current_session))):
            #     print '\t', name
            for i in range (5):
                sourcefile_emotion = sorted(glob.glob("/home/liutao/landmark-FER-SVM/CK+/cohn-kanade-images//%s//%s//*" %(part, current_session)))[-(i+1)]
                #print(sourcefile_emotion)
                #get path for last five image in sequence, which contains the emotion
                dest_emot = "/home/liutao/landmark-FER-SVM/sorted_CK+//%s//%s" %(emotions[emotion], sourcefile_emotion[-21:]) #Do same for emotion containing image
                copyfile(sourcefile_emotion, dest_emot) #Copy file

            sourcefile_neutral = sorted(glob.glob("/home/liutao/landmark-FER-SVM/CK+/cohn-kanade-images//%s//%s//*" %(part, current_session)))[0] 
            #do same for neutral image
            dest_neut = "/home/liutao/landmark-FER-SVM/sorted_CK+//neutral//%s" %sourcefile_neutral[-21:] #Generate path to put neutral image
            copyfile(sourcefile_neutral, dest_neut) #Copy file
            