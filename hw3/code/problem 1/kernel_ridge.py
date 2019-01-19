import numpy as np 
import csv 
import matplotlib.pyplot as plt 
import math

#function to calculate kernel(a,b) assuming a and b both are scalars
def calculate(a,b):
	return math.exp(-0.1*(a-b)*(a-b))

#function to do ridge regression based on the hyperparameter of ridge-reg. Plots the output for each case

def func(hyperparameter):
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


	kernel_matrix = np.zeros(shape=(num_train,num_train))

	#calculate the kernel matrix where each entry in matrix denotes the corresponding kernel similarity between two input features. 

	for i in range(num_train):
		for j in range(num_train):
			kernel_matrix[i][j] = calculate(feature[i],feature[j])


	#calculate the alpha vector of size N*1 since the weight vector is effectively a summation of each alpha multiplied by corresponding D-dimensional input vector
	temp_matrix = kernel_matrix + hyperparameter*np.identity(n=num_train,dtype=float)
	alpha = np.matmul(np.linalg.inv(temp_matrix),value)

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

	#calculate the predicted value of each input by summing over kernelised similarity between every train input and corresponding value of alpha
	for i in range(num_test):
		ans = 0
		for j in range(num_train):
			ans += alpha[j]*calculate(feature[j],test_feature[i])

		predicted_value[i] = ans
		error += (ans-test_value[i])*(ans-test_value[i])


	#calculate error and plot the scatter plots
	error = error/num_test
	print "RSME for hyperparameter value ",str(hyperparameter)," is ",str(math.sqrt(error))

	plt.scatter(test_feature,test_value,c='blue')
	plt.scatter(test_feature,predicted_value,c='red')
	plt.savefig('kernel_ridge_'+str(hyperparameter)+'.png')
	plt.clf()


#do this for every hyperparameter
func(0.1)
func(1)
func(10)
func(100)

