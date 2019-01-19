import numpy as np 
import csv 
import matplotlib.pyplot as plt 
import math

#open the data file
with open('kmeans_data.txt','r') as file:
	reader = csv.reader(file)
	raw_data = list(reader)

num_train = len(raw_data)

feature1 = np.zeros(shape=(num_train,1))
feature2 = np.zeros(shape=(num_train,1))
transformed = np.zeros(shape=(num_train,1))
cluster = np.zeros(shape=(num_train,1))

#process the data to calculate the features of each data point.
#We have transformed each data point to a 1-D point (sqrt(x**2+y**2)), or effectively to (sqrt(x**2+y**2),0). This transformation separates the data into two linearly separable regions
for i in range(num_train):
	temp = raw_data[i][0].split()
	feature1[i] = float(temp[0])
	feature2[i] = float(temp[1])
	transformed[i] = float(temp[0])**2 + float(temp[1])**2

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


plt.scatter(cluster1_x,cluster1_y,c='b')
plt.scatter(cluster2_x,cluster2_y,c='r')
plt.savefig('cluster_hand.png')
