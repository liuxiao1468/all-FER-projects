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
# get the data from the local file

data = pd.read_csv("/home/leo/deeplearning/fer2013/fer2013.csv")
size=data.shape
print ('The original shape of the data is ', size)

def get_training_data(data):
	# with open("modified_data.csv", 'wb') as f:
	#f=io.StringIO()
	#writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
	#writer = csv.writer(f, delimiter = ' ')
	x_0 = data.loc[(data['emotion']==0) & (data['Usage']== 'Training')]
	#x_0 = x_0.ix[:436,:]
	x_0 = x_0.iloc[:436,:]
	# print (x_0.head)
	# size=x_0.shape
	# print ('The shape of the data is', size)
	x_1 = data.loc[(data['emotion']==1) & (data['Usage']== 'Training')]
	x_1 = x_1.iloc[:436,:]
	#print (x_1.head)
	x_2 = data.loc[(data['emotion']==2) & (data['Usage']== 'Training')]
	x_2 = x_2.iloc[:436,:]
	x_3 = data.loc[(data['emotion']==3) & (data['Usage']== 'Training')]
	x_3 = x_3.iloc[:436,:]
	x_4 = data.loc[(data['emotion']==4) & (data['Usage']== 'Training')]
	x_4 = x_4.iloc[:436,:]
	x_5 = data.loc[(data['emotion']==5) & (data['Usage']== 'Training')]
	x_5 = x_5.iloc[:436,:]
	x_6 = data.loc[(data['emotion']==6) & (data['Usage']== 'Training')]
	x_6 = x_6.iloc[:436,:]
	Training_Data = pd.concat([x_0 , x_1, x_2, x_3, x_4, x_5, x_6])
	Training_Data.index = range (436*7)
	# size=Training_Data.shape
	# print ('The shape of the training data is', size)
	# print (Training_Data)
	#x.to_csv(modified_data.csv)
	# for row in x_0:
	# 	writer.writerow(row+'\n')
	# x_add = data.loc[ data['Usage']=='PrivateTest']
	# x_add = x_add.ix[:500,:]
	# print (x_add.head)
	# for row in x_add:
	# 	writer.writerow(row+'\n')
	return (Training_Data)

def get_testing_data(data):

	x_0 = data.loc[(data['emotion']==0) & (data['Usage']== 'PublicTest')]
	#x_0 = x_0.ix[:436,:]
	x_0 = x_0.iloc[:56,:]
	# print (x_0.head)
	# size=x_0.shape
	# print ('The shape of the data is', size)
	x_1 = data.loc[(data['emotion']==1) & (data['Usage']== 'PublicTest')]
	x_1 = x_1.iloc[:56,:]
	#print (x_1.head)
	x_2 = data.loc[(data['emotion']==2) & (data['Usage']== 'PublicTest')]
	x_2 = x_2.iloc[:56,:]
	x_3 = data.loc[(data['emotion']==3) & (data['Usage']== 'PublicTest')]
	x_3 = x_3.iloc[:56,:]
	x_4 = data.loc[(data['emotion']==4) & (data['Usage']== 'PublicTest')]
	x_4 = x_4.iloc[:56,:]
	x_5 = data.loc[(data['emotion']==5) & (data['Usage']== 'PublicTest')]
	x_5 = x_5.iloc[:56,:]
	x_6 = data.loc[(data['emotion']==6) & (data['Usage']== 'PublicTest')]
	x_6 = x_6.iloc[:56,:]
	Testing_Data = pd.concat([x_0 , x_1, x_2, x_3, x_4, x_5, x_6])
	#Testing_Data = x_0
	Testing_Data.index = range (56*7)

	return (Testing_Data)



def increase_brightness(img, value=0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def cannyEdge(img):
	minT = 140
	maxT = 255
	edge = cv2.Canny(img, minT, maxT)
	cv2.imshow("edge",edge)
	return edge

def img_preprocessing(img):
	show_image = img.reshape(48,48)
	# plt.imshow(show_image, cmap='gray')
	# plt.show()
	plt.imsave('img.png', show_image, cmap='gray')
	img = cv2.imread('img.png')
	#cv2.imshow('original img',img)
	print ('original',img)
	img = increase_brightness(img, value=50)
	# print ('after preprocessing',img)
	# cv2.imshow('img',img)
	# cv2.imwrite('img_pre1.png',img)
	edge= cannyEdge(img)
	shape=edge.shape
	# print('edges',shape)
	print('edges',edge)
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

def data_preprocessing(Training_Data):
	pixels_values = Training_Data.pixels.str.split(" ").tolist()
	pixels_values = pd.DataFrame(pixels_values, dtype=int)
	images = pixels_values.values 
	images = images.astype(np.float)
	# images_size = images.shape
	# print ('The shape of the Training images is%d', %images_size)
	for i in range (3444):
		images[i] = img_preprocessing(images[i])
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

	VALIDATION_SIZE = 392

	validation_images = images[3052:]
	validation_labels = labels[3052:]


	return images,labels,image_pixels,labels_count,validation_images,validation_labels





def CNN_training(images,labels,image_pixels,labels_count,validation_images,validation_labels):
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
	TRAINING_ITERATIONS = 10000

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
			if(392):
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











Training_Data = get_training_data(data)
#print (Training_Data.head)
size_Training_Data = Training_Data.shape
print ('The shape of the training data is', size_Training_Data)
Testing_Data = get_testing_data(data)
#print (Testing_Data.head)
size_Testing_Data = Testing_Data.shape
#print (Testing_Data.head)
print ('The shape of the testing data is', size_Testing_Data)

data_for_training = pd.concat([Training_Data, Testing_Data])
data_for_training.index = range (3444)


[images, labels, image_pixels, labels_count, validation_images, validation_labels] = data_preprocessing(data_for_training)


CNN_training(images,labels,image_pixels,labels_count,validation_images,validation_labels)