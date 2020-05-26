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

test_dataset = "/home/leo/Facial-expression-real-time/Facial-expression-real-time/sorted_CK+//%s//*" 
train_dataset = "/home/leo/Facial-expression-real-time/Facial-expression-real-time/sorted_CK+//%s//*"
# test_dataset = "/home/leo/Facial-expression-real-time/Facial-expression-real-time/MUG_dataset//%s//*" 
# train_dataset = "/home/leo/Facial-expression-real-time/Facial-expression-real-time/MUG_dataset//%s//*" 


def selection_function(individual, training_data,w1):
	AU_matrix = training_data[:,list(range(0, 228))]*(1-w1)
	VL_matrix1 = training_data[:,list(range(228, 296))]*w1
	VL_matrix2 = training_data[:,list(range(296, 364))]*w1
	index = [i for i in range(len(individual)) if individual[i] == 1]
	for i in range (len(index)):
		if index[i]<= 23:
			temp = AU_matrix[:,[ index[i]*4,index[i]*4+1,index[i]*4+2,index[i]*4+3,index[i]*4+4]]
		else:
			temp = AU_matrix[:,[ index[i]*3,index[i]*3+1,index[i]*3+2,index[i]*3+3]]

		temp1 = VL_matrix1[:,index[i]]
		temp2 = VL_matrix2[:,index[i]]

		if i == 0:
			train_1 = temp
			train_2 = temp1
			train_3 = temp2
		else:
			train_1 = np.column_stack([train_1,temp])
			train_2 = np.column_stack([train_2,temp1])
			train_3 = np.column_stack([train_3,temp2])
	train = np.column_stack([train_1,train_2])
	train = np.column_stack([train,train_3])
	return train


def apply_function(individual):
	C = cc[individual[-2]]
	w1 = ww1[individual[-1]]
	individual = individual[:68]
	train = selection_function(individual,training_data,w1)
	test = selection_function(individual,prediction_data,w1)

	clf = SVC(C = C, kernel='linear', probability=True, tol=1e-3)

	clf.fit(train, training_labels)
	pred_lin = clf.score(test, prediction_labels)
	# print ("accuracy: ", pred_lin)
	return pred_lin

import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
def scatter3d(x,y,z, cs, colorsMap='jet'):
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, c=scalarMap.to_rgba(cs))
    scalarMap.set_array(cs)
    fig.colorbar(scalarMap)
    plt.show()

# # # if you need to read ck+ 7-class
# training_data = open("training_data.pkl","rb")
# training_data = pickle.load(training_data)
# training_labels = open("training_labels.pkl","rb")
# training_labels = pickle.load(training_labels)

# prediction_data = open("prediction_data.pkl","rb")
# prediction_data = pickle.load(prediction_data)
# prediction_labels = open("prediction_labels.pkl","rb")
# prediction_labels = pickle.load(prediction_labels)

# validation_data = open("validation_data.pkl","rb")
# validation_data = pickle.load(validation_data)
# validation_labels = open("validation_labels.pkl","rb")
# validation_labels = pickle.load(validation_labels)



# save_gen = open("gen_selection.pkl","rb")
# save_gen = pickle.load(save_gen)

# save_score = open("gen_score.pkl","rb")
# save_score = pickle.load(save_score)


# acc_avg = open('acc_avg.pkl','rb')
# acc_avg = pickle.load(acc_avg)




# # if you need to read ck+ 8-class
# training_data = open("training_data1.pkl","rb")
# training_data = pickle.load(training_data)
# training_labels = open("training_labels1.pkl","rb")
# training_labels = pickle.load(training_labels)

# prediction_data = open("prediction_data1.pkl","rb")
# prediction_data = pickle.load(prediction_data)
# prediction_labels = open("prediction_labels1.pkl","rb")
# prediction_labels = pickle.load(prediction_labels)

# validation_data = open("validation_data1.pkl","rb")
# validation_data = pickle.load(validation_data)
# validation_labels = open("validation_labels1.pkl","rb")
# validation_labels = pickle.load(validation_labels)



# save_gen = open("gen_selection1.pkl","rb")
# save_gen = pickle.load(save_gen)

# save_score = open("gen_score1.pkl","rb")
# save_score = pickle.load(save_score)


# acc_avg = open('acc_avg1.pkl','rb')
# acc_avg = pickle.load(acc_avg)


# # if you need to read MUG 7-class
training_data = open("training_data3.pkl","rb")
training_data = pickle.load(training_data)
training_labels = open("training_labels3.pkl","rb")
training_labels = pickle.load(training_labels)

prediction_data = open("prediction_data3.pkl","rb")
prediction_data = pickle.load(prediction_data)
prediction_labels = open("prediction_labels3.pkl","rb")
prediction_labels = pickle.load(prediction_labels)

validation_data = open("validation_data3.pkl","rb")
validation_data = pickle.load(validation_data)
validation_labels = open("validation_labels3.pkl","rb")
validation_labels = pickle.load(validation_labels)



save_gen = open("gen_selection3.pkl","rb")
save_gen = pickle.load(save_gen)

save_score = open("gen_score3.pkl","rb")
save_score = pickle.load(save_score)


acc_avg = open('acc_avg3.pkl','rb')
acc_avg = pickle.load(acc_avg)


cc = [0.1, 1, 10, 100, 316, 1000, 3162, 10000]
ww1 = np.arange(0, 1, 0.05)

# for i in range (len(save_gen)):
# 	print(i, "", acc_avg[i], " ",cc[save_gen[i][-2]], " ", ww1[save_gen[i][-1]])


index = acc_avg.index(max(acc_avg))
# index = 295
fitness_value = save_score[index]
print("Best",fitness_value, " ", index)
print(save_gen[index])
select = save_gen[index]
# C = 100
pred_lin = apply_function(select)
print("final result: ", pred_lin)

# clf = SVC(C = C, kernel='linear', probability=True, tol=1e-3)

# clf.fit(train, training_labels)
# pred_lin = clf.score(test, prediction_labels)
# print()
# print (pred_lin)



# for i in range (len(save_gen)):
#     select = save_gen[i]
#     C = 100
#     train = selection_function(select,training_data)
#     test = selection_function(select,prediction_data)

#     clf = SVC(C = C, kernel='linear', probability=True, tol=1e-3)

#     clf.fit(train, training_labels)
#     pred_lin = clf.score(test, prediction_labels)
#     print (pred_lin)

plot_w1 = []
plot_c = []
plot_acc = []
for i in range (len(save_gen)):
	plot_w1.append(ww1[save_gen[i][-1]])
	plot_c.append(cc[save_gen[i][-2]])
	plot_acc.append(acc_avg[i])

# Create Map
cm = plt.get_cmap("RdYlGn")

x = np.asarray(plot_w1)
y = np.asarray(plot_c)
z = np.asarray(plot_acc)
col = np.arange(0.0033,0.99,0.0033)



# 3D Plot
fig = plt.figure()
ax3D = fig.add_subplot(111, projection='3d')
p3d = ax3D.scatter(x, y, z, s=30, c=col, marker='o')
ax3D.set_title('Accuracy with varied Weights and C value')
ax3D.set_xlabel('w1')
ax3D.set_ylabel('C Value')
ax3D.set_zlabel('Accuracy')   
fig.colorbar(p3d, shrink=0.4, aspect=5)                                                                          

plt.show()










# select = [0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1]

# C = 100
# train = selection_function(select,training_data)
# test = selection_function(select,prediction_data)

# clf = SVC(C = C, kernel='linear', probability=True, tol=1e-3)

# clf.fit(train, training_labels)
# pred_lin = clf.score(test, prediction_labels)
# print ("accuracy: ", pred_lin)
############################################################### training and plot Confusion Matrix #################################


# accur_lin = []

# w1 = 0.75
# w2 = 1-w1
# C = 100

# print("right now w1 is:", w1)

# clf = SVC(C = C, kernel='linear', probability=True, tol=1e-3)
# accuracy_inside = []

# for m in range(0,1):
#     # print("Making sets %s" %m) #Make sets by random sampling 80/20%
#     [training_data, training_labels, prediction_data, prediction_labels] = make_training_sets(w1,w2)
#     # print("training SVM linear %s" %m) #train SVM
#     clf.fit(training_data, training_labels)
#     print("getting accuracies %s" %m) #Use score() function to get accuracy
#     pred_lin = clf.score(prediction_data, prediction_labels)
#     print ("linear: ", pred_lin)
#     accuracy_inside.append(pred_lin) #Store accuracy in a list
#     joblib.dump(clf, 'real_time_best_landmark_SVM.pkl')
# accur_lin.append(accuracy_inside)

# class_names = ["anger",  "disgust" ,"fear","happiness", "sadness","surprise","neutral"]


# clf = joblib.load('real_time_best_landmark_SVM.pkl') 
# pridict_pridction_labels = clf.predict(prediction_data)

# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)

#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.tight_layout()


# # Compute confusion matrix
# cnf_matrix = confusion_matrix(prediction_labels, pridict_pridction_labels)
# np.set_printoptions(precision=2)

# # Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names,
#                       title='Confusion matrix, without normalization')

# # Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')

# plt.show()

