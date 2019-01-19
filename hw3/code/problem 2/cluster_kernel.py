import numpy as np 
import csv 
import matplotlib.pyplot as plt 
import math

#function to calculate kernel(a,b) assuming a and b both are lists of same dimension, denoting two input vectors
def calculate(a,b):
	return math.exp(-0.1*((np.linalg.norm(np.asarray(a)-np.asarray(b)))**2) )

#function to do clustering, trial denotes the number of times it has been invoked
def func(trial):
	#open the train file
	with open('kmeans_data.txt','r') as file:
		reader = csv.reader(file)
		raw_data = list(reader)

	num_train = len(raw_data)

	feature1 = np.zeros(shape=(num_train,1))
	feature2 = np.zeros(shape=(num_train,1))
	transformed = np.zeros(shape=(num_train,1))
	cluster = np.zeros(shape=(num_train,1))

	#process the data to have both features in arrays
	for i in range(num_train):
		temp = raw_data[i][0].split()
		feature1[i] = float(temp[0])
		feature2[i] = float(temp[1])
		
	#random index to be chosen as the landmark point. 
	index= np.random.randint(0,num_train)

	#since we have chosen only 1 landmark, each input vector is now converted to a scalar value (1-D vector) based on kernel value with the random landmark pouint
	for i in range(num_train):
		transformed[i] = calculate(a=[feature1[index],feature2[index]],b=[feature1[i],feature2[i]])

	mean1 = transformed[0]
	mean2 = transformed[1]
	cluster[0] = 0
	cluster[1] =1
	prev_mean1 = mean1
	prev_mean2 = mean2

	#clustering algo
	while(1):

		#calculate the cluster id of each data point based on current means of clustes
		for i in range(num_train):
			if((transformed[i]-mean1)**2 > (transformed[i]-mean2)**2 ):
				cluster[i] = 1
			else:
				cluster[i] = 0

		temp1 = 0
		temp2 = 0
		num_1 = 0
		num_2 = 0

		#calculate the mean of both clusters based on new assignment of cluster ids
		for i in range(num_train):
			if(cluster[i] == 0):
				temp1 += transformed[i]
				num_1 += 1
			else:
				temp2 += transformed[i]
				num_2 += 1
		
		prev_mean1 = mean1
		prev_mean2 = mean2
		mean1 = temp1/num_1
		mean2 = temp2/num_2

		#if means don't change any further, we have converged. 
		if(int(mean1) == int(prev_mean1) and int(mean2) == int(prev_mean2) ):
			break

	cluster1_x =[]
	cluster1_y =[]
	cluster2_x = []
	cluster2_y = []

	#based on each cluster, plot the points in different colors. 
	for i in range(num_train):
		if(cluster[i] == 0):
			cluster1_x.append(feature1[i])
			cluster1_y.append(feature2[i])
		else:
			cluster2_x.append(feature1[i])
			cluster2_y.append(feature2[i])


	plt.scatter(cluster1_x,cluster1_y,c='green')
	plt.scatter(cluster2_x,cluster2_y,c='red')
	plt.scatter(feature1[index],feature2[index],c='blue')

	plt.savefig('cluster_kernel_'+str(trial)+'.png')
	plt.clf()


for i in range(10):
	func(i)


