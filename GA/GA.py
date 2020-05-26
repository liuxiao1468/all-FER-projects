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
from random import randrange

test_dataset = "/home/leo/Facial-expression-real-time/Facial-expression-real-time/sorted_CK+//%s//*" 
train_dataset = "/home/leo/Facial-expression-real-time/Facial-expression-real-time/sorted_CK+//%s//*"
# test_dataset = "/home/leo/Facial-expression-real-time/Facial-expression-real-time/MUG_dataset//%s//*" 
# train_dataset = "/home/leo/Facial-expression-real-time/Facial-expression-real-time/MUG_dataset//%s//*" 

cc = [0.1, 1, 10, 100, 316, 1000, 3162, 10000]
ww1 = np.arange(0, 1, 0.05)
# print(randrange(8))
# print(len(w1))

def generate_population(size):

    population = []
    for i in range(size):
        zero_count = random.randint(0, 68)
        one_count = 68 - zero_count
        individual = [0]*zero_count + [1]*one_count
        random.shuffle(individual)
        individual.append(randrange(8))
        individual.append(randrange(20))
        population.append(individual)

    return population


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
	test = selection_function(individual,validation_data,w1)

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
	c1 = individual_a[-2]
	c2 = individual_b[-2]
	w1_a = individual_a[-1]
	w1_b = individual_b[-1]
	individual_a = individual_a[:68]
	individual_b = individual_b[:68]


	gen_1 = individual_a[int(len(individual_a)/2) :]
	gen_2 = individual_b[: int(len(individual_b)/2)]

	new_individual = gen_1 + gen_2
	draw1 = random.uniform(0, 1)
	if draw1 <= 0.50:
		c = c1
		w1 = w1_b
	else:
		c = c2
		w1 = w1_a		
	# if draw1 < 0.33:
	# 	c = math.floor((c1+c2)/2)
	# 	w1 = math.floor((w1_a+w1_b)/2)
	# elif (draw1 > 0.33) and (draw1 < 0.66) :
	# 	c = c1
	# 	w1 = w1_b
	# else:
	# 	c = c2
	# 	w1 = w1_a
	new_individual.append(c)
	new_individual.append(w1)
	return new_individual


def mutate(individual):
	draw1 = random.uniform(0, 1)
	draw2 = random.uniform(0, 1)
	if draw1>0.97:
		p = random.choice([-1,1])
		if (individual[-2]+p > 0) and (individual[-2]+p < 7):
			individual[-2] = individual[-2]+p
	if draw2>0.9:
		p = random.choice([-2,2])
		if (individual[-1]+p > 0) and (individual[-1]+p < 20):
			individual[-1] = individual[-1]+p

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
    print(sum(fitness)/len(fitness), " ", cc[sorted_by_fitness_population[-1][-2]])
    acc_avg.append(sum(fitness)/len(fitness))
    acc_up.append(fitness[-1]-(sum(fitness)/len(fitness)))
    acc_low.append((sum(fitness)/len(fitness))-fitness[0])
    temp = []
    for i in range(population_size):
    	temp.append(sum(sorted_by_fitness_population[i][:68]))
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



# # if you need to read for 7 class ck+
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

# # if you need to read for 8 class ck+
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

# if you need to read for 7 class MUG
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



save_gen = []
save_score = []

acc_avg = []
acc_up = []
acc_low = []

num_avg = []
num_up = []
num_low = []

population = generate_population(size=10)
generations = 300
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


# # # save files for 7-class ck+
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




# # save files for 8-class ck+
# with open('gen_selection1.pkl','wb') as f:
#     pickle.dump(save_gen, f)

# with open('gen_score1.pkl','wb') as f:
#     pickle.dump(save_score, f)

# with open('acc_avg1.pkl','wb') as f:
#     pickle.dump(acc_avg, f)
# with open('acc_up1.pkl','wb') as f:
#     pickle.dump(acc_up, f)
# with open('acc_low1.pkl','wb') as f:
#     pickle.dump(acc_low, f)


# with open('num_avg1.pkl','wb') as f:
#     pickle.dump(num_avg, f)
# with open('num_up1.pkl','wb') as f:
#     pickle.dump(num_up, f)
# with open('num_low1.pkl','wb') as f:
#     pickle.dump(num_low, f)


# # save files for 7-class MUG
with open('gen_selection3.pkl','wb') as f:
    pickle.dump(save_gen, f)

with open('gen_score3.pkl','wb') as f:
    pickle.dump(save_score, f)

with open('acc_avg3.pkl','wb') as f:
    pickle.dump(acc_avg, f)
with open('acc_up3.pkl','wb') as f:
    pickle.dump(acc_up, f)
with open('acc_low3.pkl','wb') as f:
    pickle.dump(acc_low, f)


with open('num_avg3.pkl','wb') as f:
    pickle.dump(num_avg, f)
with open('num_up3.pkl','wb') as f:
    pickle.dump(num_up, f)
with open('num_low3.pkl','wb') as f:
    pickle.dump(num_low, f)

print("---Save the result into a pickle file---")