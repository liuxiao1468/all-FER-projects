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


def generate_population(size):

    population = []
    for i in range(size):
        zero_count = random.randint(0, 68)
        one_count = 68 - zero_count
        individual = [0]*zero_count + [1]*one_count
        random.shuffle(individual)
        population.append(individual)

    return population


def selection_function(individual, training_data):
	AU_matrix = training_data[:,list(range(0, 228))]
	VL_matrix1 = training_data[:,list(range(228, 296))]
	VL_matrix2 = training_data[:,list(range(296, 364))]
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
	C = 100
	train = selection_function(individual,training_data)
	test = selection_function(individual,validation_data)

	clf = SVC(C = C, kernel='linear', probability=True, tol=1e-3)

	clf.fit(train, training_labels)
	pred_lin = clf.score(test, validation_labels)
	# print ("accuracy: ", pred_lin)
	return pred_lin

def sort_population_by_fitness(population):
	fitness = []
	for i in range (len(population)):
		fitness_score = apply_function(population[i])
		fitness.append(fitness_score)
	population = [x for _,x in sorted(zip(fitness, population))]
	fitness = sorted(fitness)
	return population, fitness


def choice_by_roulette(sorted_population, fitness):
    offset = 0
    normalized_fitness_sum = sum(fitness)

    lowest_fitness = fitness[0]
    draw = random.uniform(0, 1)

    accumulated = 0
    for i in range(len(sorted_population)):
        probability = fitness[i] / normalized_fitness_sum
        accumulated += probability

        if draw <= accumulated:
            return sorted_population[i]


def crossover(individual_a, individual_b):
	gen_1 = individual_a[int(len(individual_a)/2) :]
	gen_2 = individual_b[: int(len(individual_b)/2)]

	new_individual = gen_1 + gen_2
	return new_individual


def mutate(individual):
	for i in range (3):
		idx = random.randint(0, 67)
		if individual[idx] == 1:
			individual[idx] = 0
		else:
			individual[idx] = 1
	return individual


def make_next_generation(previous_population):
    next_generation = []
    sorted_by_fitness_population, fitness = sort_population_by_fitness(previous_population)
    population_size = len(previous_population)
    print(sum(fitness)/len(fitness))
    acc_avg.append(sum(fitness)/len(fitness))
    acc_up.append(fitness[-1]-(sum(fitness)/len(fitness)))
    acc_low.append((sum(fitness)/len(fitness))-fitness[0])
    temp = []
    for i in range(population_size):
    	temp.append(sum(sorted_by_fitness_population[i]))
    num_avg.append(sum(temp)/len(temp))
    num_up.append(max(temp)-(sum(temp)/len(temp)))
    num_low.append((sum(temp)/len(temp))-min(temp))

    # print("fitness score:")
    # for i in range (population_size):
    # 	print(fitness[i])
    save_gen.append(sorted_by_fitness_population[-1])
    save_score.append(fitness[-1])
    for i in range(population_size):
	    first_choice = choice_by_roulette(sorted_by_fitness_population, fitness)
	    second_choice = choice_by_roulette(sorted_by_fitness_population, fitness)
	    individual = crossover(first_choice, second_choice)
	    individual = mutate(individual)
	    next_generation.append(individual)
    return next_generation



# # if you need to read for 7 class
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

# if you need to read for 6 class
training_data = open("training_data1.pkl","rb")
training_data = pickle.load(training_data)
training_labels = open("training_labels1.pkl","rb")
training_labels = pickle.load(training_labels)

prediction_data = open("prediction_data1.pkl","rb")
prediction_data = pickle.load(prediction_data)
prediction_labels = open("prediction_labels1.pkl","rb")
prediction_labels = pickle.load(prediction_labels)

validation_data = open("validation_data1.pkl","rb")
validation_data = pickle.load(validation_data)
validation_labels = open("validation_labels1.pkl","rb")
validation_labels = pickle.load(validation_labels)



save_gen = []
save_score = []

acc_avg = []
acc_up = []
acc_low = []

num_avg = []
num_up = []
num_low = []

population = generate_population(size=10)
generations = 100
i = 1
while True:
    # print(f"🧬 GENERATION {i}")
    if i == generations:
        break
    i += 1
    population = make_next_generation(population)
    # print()

best_individual, fitness = sort_population_by_fitness(population)
print("\n🔬 FINAL RESULT")
print(best_individual[-1])
print(fitness[-1])


# with open('gen_selection.pkl','wb') as f:
#     pickle.dump(save_gen, f)

# with open('gen_score.pkl','wb') as f:
#     pickle.dump(save_score, f)

# with open('acc_avg.pkl','wb') as f:
#     pickle.dump(acc_avg, f)
# with open('acc_up.pkl','wb') as f:
#     pickle.dump(acc_up, f)
# with open('acc_low.pkl','wb') as f:
#     pickle.dump(acc_low, f)


# with open('num_avg.pkl','wb') as f:
#     pickle.dump(num_avg, f)
# with open('num_up.pkl','wb') as f:
#     pickle.dump(num_up, f)
# with open('num_low.pkl','wb') as f:
#     pickle.dump(num_low, f)

# print("---Save the result into a pickle file---")




# save files for 6-class
with open('gen_selection1.pkl','wb') as f:
    pickle.dump(save_gen, f)

with open('gen_score1.pkl','wb') as f:
    pickle.dump(save_score, f)

with open('acc_avg1.pkl','wb') as f:
    pickle.dump(acc_avg, f)
with open('acc_up1.pkl','wb') as f:
    pickle.dump(acc_up, f)
with open('acc_low1.pkl','wb') as f:
    pickle.dump(acc_low, f)


with open('num_avg1.pkl','wb') as f:
    pickle.dump(num_avg, f)
with open('num_up1.pkl','wb') as f:
    pickle.dump(num_up, f)
with open('num_low1.pkl','wb') as f:
    pickle.dump(num_low, f)











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

