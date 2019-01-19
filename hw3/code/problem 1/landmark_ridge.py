import numpy as np 
import csv 
import matplotlib.pyplot as plt 
import math

#function to calculate kernel(a,b) assuming a and b both are scalars
def calculate(a,b):
	return math.exp(-0.1*(a-b)*(a-b))


#function to do ridge regression based on the number of landmark points chosen. Plots the output for each case
def func(landmarks):
	#read the input file
	with open('ridgetrain.txt','r') as file:
		reader = csv.reader(file)
		raw_data = list(reader)

	num_train = len(raw_data)

	feature = np.zeros(shape=(num_train,1))
	value = np.zeros(shape=(num_train,1))

	#calculate the feature of every input, and its corresponding output. Here, we had to split each row of csv reader since it was not separated through comma and was a string
	for i in range(num_train):
		temp = raw_data[i][0].split()
		feature[i] = float(temp[0])
		value[i] = float(temp[1])

	landmarks_list = []

	feature_matrix = np.zeros(shape=(num_train,landmarks))

	#calculate the landmark points randomly from dataset using randint, since each input has only one feature, we can form a list of all landmarks points where each entry is 1-D landmark point
	for i in range(landmarks):
		landmarks_list.append(feature[np.random.randint(0,num_train)])

	#based on landmark points, calculate the feature vector of every train input of L-dimensional, using kernel similarity with each landmark point.
	for i in range(num_train):
		for j in range(landmarks):
			feature_matrix[i][j] = calculate(feature[i],landmarks_list[j])

	weight_vector = np.matmul(np.transpose(feature_matrix),np.matmul(np.linalg.inv(0.1*np.identity(n=num_train,dtype=float)+np.matmul(feature_matrix,np.transpose(feature_matrix))),value))
	
	#open the test file
	with open('ridgetest.txt','r') as file:
		reader = csv.reader(file)
		raw_data = list(reader)

	num_test = len(raw_data)

	test_feature = np.zeros(shape=(num_test,1))
	test_value = np.zeros(shape=(num_test,1))

	#process the data and extract features and value for test file
	for i in range(num_test):
		temp = raw_data[i][0].split()
		test_feature[i] = float(temp[0])
		test_value[i] = float(temp[1])

	predicted_value = np.zeros(shape=(num_test,1))
	error = 0.0


	for i in range(num_test):
		temp = np.zeros(shape=(landmarks,1),dtype=float)
		
		#calculate the L-dimensional feature representation of test input
		for j in range(landmarks):
			temp[j] = calculate(test_feature[i],landmarks_list[j])

		#based on new feature rep, calculate the value using simple linear model
		ans = np.matmul(np.transpose(weight_vector),temp)
		error += (ans-test_value[i])*(ans-test_value[i])
		predicted_value[i] = ans


	#calculate the errors and plot the outputs
	error = error/num_test
	print "RSME for landmark value ",str(landmarks)," is ",str(math.sqrt(error))

	plt.scatter(test_feature,test_value,c='blue')
	plt.scatter(test_feature,predicted_value,c='red')
	plt.savefig('landmark_ridge_'+str(landmarks)+'.png')
	plt.clf()


#do this for all values of landmarks
func(2)
func(5)
func(10)
func(20)
func(50)
func(100)

