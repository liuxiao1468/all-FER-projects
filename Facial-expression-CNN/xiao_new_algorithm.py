import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import csv
import io
from random import *
# get the data from the local file

data = pd.read_csv("/home/leo/deeplearning/fer2013/fer2013.csv")
size=data.shape
print ('The original shape of the data is ', size)

def get_all_data(data):
	x_0 = data.loc[(data['Usage']== 'Training')]
	size_training=x_0.shape
	x_1 = data.loc[(data['Usage']== 'PrivateTest')]
	size_testing=x_1.shape
	print('The Training shape of the data is ', size_training)
	print('The Testing shape of the data is',size_testing)
	all_data = pd.concat([x_0 , x_1])
	all_data.index = range(32298)
	return all_data



def increase_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def cannyEdge(img,minT,maxT):
	# minT = 80
	# maxT = 100
	edge = cv2.Canny(img, minT, maxT)
	cv2.imshow("edge",edge)
	return edge

def img_preprocessing(img,PF):
	show_image = img.reshape(48,48)
	# plt.imshow(show_image, cmap='gray')
	# plt.show()
	plt.imsave('img.png', show_image, cmap='gray')
	img = cv2.imread('img.png')
	#cv2.imshow('original img',img)
	#print ('original',img)
	img = increase_brightness(img, value=PF[0])
	# print ('after preprocessing',img)
	# cv2.imshow('img',img)
	# cv2.imwrite('img_pre1.png',img)
	edge= cannyEdge(img, minT=PF[1], maxT=PF[2])
	shape=edge.shape
	# print('edges',shape)
	#print('edges',edge)
	edge = edge.reshape(-1)
	# cv2.imwrite('img_pre2.png',edge)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	return (edge)


def labels_one_hot_encoding(labels_dense, num_classes):
	num_labels = labels_dense.shape[0]
	index_offset = np.arange(num_labels)*num_classes
	labels_one_hot = np.zeros((num_labels, num_classes))
	labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
	return (labels_one_hot)

def data_preprocessing(Training_Data,random_x):
	pixels_values = Training_Data.pixels.str.split(" ").tolist()
	pixels_values = pd.DataFrame(pixels_values, dtype=int)
	images = pixels_values.values 
	images = images.astype(np.float)
	
	ini_bright = random_x[0]
	ini_edge_min = random_x[1]
	ini_edge_max = random_x[2]


	bright_contrast=randint(-60,140)
	edge_min=randint(-100,100)
	edge_max=randint(-100,100)

	while (ini_edge_max+edge_max<ini_edge_min+edge_min) and (ini_bright+bright_contrast<0):
		bright_contrast=randint(-60,140)
		edge_min=randint(-100,100)
		edge_max=randint(-100,100)

	PF = np.array([ini_bright+bright_contrast, ini_edge_min+edge_min, ini_edge_max+edge_max], dtype=np.uint8)
	# images_size = images.shape
	# print ('The shape of the Training images is%d', %images_size)
	# for i in range (32298):
	# 	images[i] = img_preprocessing(images[i],PF)
	#images[images>0] = 1
	images_size = images.shape
	print ('The shape of the Training images is', images_size)
	image_pixels = images.shape[1]
	image_width = image_height = np.ceil(np.sqrt(image_pixels)).astype(np.uint8)

	labels_flat = Training_Data["emotion"].values.ravel()
	labels_count = np.unique(labels_flat).shape[0]
	print ('The number of different facial expressions is',labels_count)
	print ('The number of labels is',labels_flat.shape[0])

	labels = labels_one_hot_encoding(labels_flat, labels_count)
	labels = labels.astype(np.uint8)

	VALIDATION_SIZE = 3589

	validation_images = images[28709:]
	validation_labels = labels[28709:]


	return images,labels,image_pixels,labels_count,validation_images,validation_labels,PF





def CNN_training(images,labels,image_pixels,labels_count,validation_images,validation_labels):
	tf.reset_default_graph()
	# weight initialization
	def weight_variable(shape):
		initial = tf.truncated_normal(shape, stddev=1e-4)
		return tf.Variable(initial)

	def bias_variable(shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)
	#convolution
	def conv2d(x, W, padd):
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padd)
	# pooling
	def max_pool_2x2(x):
		return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

	# images
	x = tf.placeholder('float', shape=[None, image_pixels])
	# labels
	y_ = tf.placeholder('float', shape=[None, labels_count])

	#build first convolution layer 
	W_conv1 = weight_variable([5, 5, 1, 64])
	b_conv1 = bias_variable([64])

	image = tf.reshape(x, [-1 , 48, 48, 1])

	h_conv1 = tf.nn.relu(conv2d(image, W_conv1, "SAME") + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)


	# second convolutional layer
	W_conv2 = weight_variable([5, 5, 64, 128])
	b_conv2 = bias_variable([128])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, "SAME") + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	# local layer weight initialization
	def local_weight_variable(shape):
		initial = tf.truncated_normal(shape, stddev=0.04)
		return tf.Variable(initial)

	def local_bias_variable(shape):
		initial = tf.constant(0.0, shape=shape)
		return tf.Variable(initial)

	# densely connected layer local 3
	W_fc1 = local_weight_variable([12 * 12 * 128, 3072])
	b_fc1 = local_bias_variable([3072])

	h_pool2_flat = tf.reshape(h_pool2, [-1, 12 * 12 * 128])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	# densely connected layer local 4
	W_fc2 = local_weight_variable([3072, 1536])
	b_fc2 = local_bias_variable([1536])

	h_fc2_flat = tf.reshape(h_fc1, [-1, 3072])
	h_fc2 = tf.nn.relu(tf.matmul(h_fc2_flat, W_fc2) + b_fc2)

	# dropout
	keep_prob = tf.placeholder('float')
	h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

	# readout layer for deep net
	W_fc3 = weight_variable([1536, labels_count])
	b_fc3 = bias_variable([labels_count])

	y = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

	LEARNING_RATE = 1e-4
	# cost function
	cross_entropy = -tf.reduce_sum(y_*tf.log(y))
	# optimisation function
	train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
	# evaluation
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

	accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
	predict = tf.argmax(y,1)
	# set to 3000 iterations 
	TRAINING_ITERATIONS = 3600

	DROPOUT = 0.5
	batch_size = 50

	epochs_completed = 0
	index_in_epoch = 0
	num_examples = images.shape[0]

    # serve data by batches

	# def next_batch(batch_size):

	# 	global images
	# 	global labels
	# 	global index_in_epoch
	# 	global epochs_completed

	# 	start = index_in_epoch
	# 	index_in_epoch += batch_size

	# 	# when all trainig data have been already used, it is reorder randomly    
	# 	if index_in_epoch > num_examples:
	# 		# finished epoch
	# 		epochs_completed += 1
	# 		# shuffle the data
	# 		perm = np.arange(num_examples)
	# 		np.random.shuffle(perm)
	# 		images = images[perm]
	# 		labels = labels[perm]
	# 		# start next epoch
	# 		start = 0
	# 		index_in_epoch = batch_size
	# 		assert batch_size <= num_examples
	# 	end = index_in_epoch
	# 	return images[start:end], labels[start:end]


	# start TensorFlow session
	init = tf.global_variables_initializer()
	# config = tf.ConfigProto()
	# config.gpu_options.allow_growth = True
	# session = tf.Session(config=config, ...)
	# config = tf.ConfigProto()
	# config.gpu_options.allocator_type ='BFC'
	# config.gpu_options.per_process_gpu_memory_fraction = 0.90
	sess = tf.InteractiveSession()

	sess.run(init)

	# visualisation variables
	train_accuracies = []
	validation_accuracies = []
	x_range = []

	display_step=1

	for i in range(TRAINING_ITERATIONS):
		#get new batch
		#batch_xs, batch_ys = next_batch(BATCH_SIZE)
		# global images
		# global labels
		# global index_in_epoch
		# global epochs_completed

		start = index_in_epoch
		index_in_epoch += batch_size

		# when all trainig data have been already used, it is reorder randomly    
		if index_in_epoch > num_examples:
			# finished epoch
			epochs_completed += 1
			# shuffle the data
			perm = np.arange(num_examples)
			np.random.shuffle(perm)
			images = images[perm]
			labels = labels[perm]
			# start next epoch
			start = 0
			index_in_epoch = batch_size
			assert batch_size <= num_examples
		end = index_in_epoch
		batch_xs = images[start:end]
		batch_ys = labels[start:end]
        
		# check progress on every 1st,2nd,...,10th,20th,...,100th... step
		if i%display_step == 0 or (i+1) == TRAINING_ITERATIONS:
			train_accuracy = accuracy.eval(feed_dict={x:batch_xs, 
		                                              y_: batch_ys, 
		                                              keep_prob: 1.0})       
			if(3589):
				validation_accuracy = accuracy.eval(feed_dict={ x: validation_images[0:batch_size], 
		                                                        y_: validation_labels[0:batch_size], 
		                                                        keep_prob: 1.0})
				print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d' %(train_accuracy, validation_accuracy,i))
				validation_accuracies.append(validation_accuracy)
			else:
				print('training_accuracy => %.4f for step %d'%(train_accuracy, i))
			train_accuracies.append(train_accuracy)
			x_range.append(i)

			if i%(display_step*10) == 0 and i and display_step<100:
				display_step *= 10
		# train on batch
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: DROPOUT})

	train_accuracies = np.asarray(train_accuracies)
	validation_accuracies = np.asarray(validation_accuracies)
	size_train = train_accuracies.shape
	size_validate = validation_accuracies.shape
	# print('train_accuracies:',train_accuracies)
	# print('the size of train_accrucies is:',size_train)
	# print('validation_accuracies:' , validation_accuracies)
	# print('the size of validation_accuracies is:',size_validate)


	# if(3589):
	# 	validation_accuracy = accuracy.eval(feed_dict={x: validation_images, y_: validation_labels, keep_prob: 1.0})
	# 	print('validation_accuracy => %.4f'%validation_accuracy)
	# 	plt.plot(x_range, train_accuracies,'-b', label='Training')
	# 	plt.plot(x_range, validation_accuracies,'-g', label='Validation')
	# 	plt.legend(loc='lower right', frameon=False)
	# 	plt.ylim(ymax = 1.0, ymin = 0.0)
	# 	plt.ylabel('accuracy')
	# 	plt.xlabel('step')
	# 	plt.show()

	# saver = tf.train.Saver(tf.global_variables())
	# saver.save(sess, "./my-model",global_step = 0)

	x1 = np.ceil(train_accuracies[:10]*100)
	area1 = np.trapz(x1,dx=1)
	x2 = np.ceil(train_accuracies[10:19]*100)
	area2 = np.trapz(x2,dx=10)
	x3= np.ceil(train_accuracies[20:54]*100)
	area3 = np.trapz(x3,dx=100)
	area = (area1+area2+area3)/100
	print('The under area of the learning curve is :',area)
	if area>2101.11:
		y=1
	else:
		y=0

	sess.close()
	print('The y value is:',y)

	return(train_accuracies,validation_accuracies,y,area)




all_data = get_all_data(data)

random_value = np.array([[10,0,255],[50,50,150],[100,100,200],[150,0,100],[200,200,255]])
number_PF = 5
for m in range (number_PF):
	X=[]
	Y=[]
	learning_area=[]
	iteration_time = 100
	random_x=random_value[m]
	for i in range (iteration_time):
		y=[]
		[images, labels, image_pixels, labels_count, validation_images, validation_labels,PF] = data_preprocessing(all_data,random_x)
		PF = [PF[0]-random_x[0],PF[1]-random_x[1],PF[2]-random_x[2]]
		X=np.append(X,PF,axis=0)
		[train_accuracies,validation_accuracies,y,area]=CNN_training(images,labels,image_pixels,labels_count,validation_images,validation_labels)
		Y=np.append(Y,y)
		learning_area=np.append(learning_area,area)
	print('The input PreFilter is:',X)
	print('The shape of input :', X.shape)
	np.savetxt('PF_value'+str(m)+'.txt', X, fmt='%d')
	print('The output score is:', Y)
	print('The shape of output is',Y.shape)
	np.savetxt('score_value'+str(m)+'.txt', Y, fmt='%d')
	print('The learning area is',learning_area)
	print('The shape of learning area is',learning_area.shape)
	np.savetxt('areas_value'+str(m)+'.txt', learning_area, fmt='%1.4e')

