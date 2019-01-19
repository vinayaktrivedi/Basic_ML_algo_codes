import numpy as np  

#load X_seen
X_seen = np.load('X_seen.npy')

#number of seen classes
seen_classes = len(X_seen)

#number of features
features = X_seen[0].shape[1]

#matrix to store mean of each seen class
mean = np.zeros(shape=(seen_classes,features),dtype=float)

#Compute mean of each seen class
for i in range(seen_classes):	

	for j in range(X_seen[i].shape[0]):
		mean[i] += X_seen[i][j]
	mean[i] = mean[i]/X_seen[i].shape[0]

#load class attributes 
class_attributes_seen = np.load('class_attributes_seen.npy')
class_attributes_unseen = np.load('class_attributes_unseen.npy')

#number of unseen classes
unseen_classes = len(class_attributes_unseen)

#matrix to store mean of unseen classes
mean_unseen = np.zeros(shape=(unseen_classes,features),dtype=float)

#load test data
Xtest = np.load('Xtest.npy')
Ytest = np.load('Ytest.npy')

#number of test inputs
input_number = len(Xtest)

identity = np.identity(class_attributes_seen.shape[1])

#for each different value of lamba, we will calculate our weight matrix, and then mean of each unseen class to make predictions
lamba = 0.01

#calculation of weight according to ridge regression formula 
weight = np.matmul(np.linalg.inv(lamba*identity + np.matmul(np.transpose(class_attributes_seen),class_attributes_seen)),np.matmul(np.transpose(class_attributes_seen),mean))

#calculation of mean of unseen classes based on weight matrix
for i in range(unseen_classes):
	mean_unseen[i] = np.matmul(class_attributes_unseen[i],weight)

#make predictions
Ypredicted = np.ndarray(shape=(input_number,1))
correct_labels = 0

for i in range(input_number):
	min_dist = float("inf")
	min_index = 0
	
	for j in range(unseen_classes):
		#euclidean distance from mean of unseen classes to test inputs
		temp = np.linalg.norm(mean_unseen[j]-Xtest[i])
		if(temp <= min_dist):
			min_dist = temp
			min_index = j+1

	Ypredicted[i] = min_index

	if(min_index == Ytest[i]):
		correct_labels += 1.0

#print accuracy
accuracy = correct_labels/input_number
print("Accuracy of Model for Lambda=0.01 is "+str(accuracy*100))

#same process repeating for every lambda thereafter

lamba = 0.1
weight = np.matmul(np.linalg.inv(lamba*identity + np.matmul(np.transpose(class_attributes_seen),class_attributes_seen)),np.matmul(np.transpose(class_attributes_seen),mean))

for i in range(unseen_classes):
	mean_unseen[i] = np.matmul(class_attributes_unseen[i],weight)

correct_labels = 0

for i in range(input_number):
	min_dist = float("inf")
	min_index = 0
	
	for j in range(unseen_classes):
		temp = np.linalg.norm(mean_unseen[j]-Xtest[i])
		if(temp <= min_dist):
			min_dist = temp
			min_index = j+1

	Ypredicted[i] = min_index

	if(min_index == Ytest[i]):
		correct_labels += 1.0


#print accuracy
accuracy = correct_labels/input_number
print("Accuracy of model for Lambda=0.1 is "+str(accuracy*100))


lamba = 1
weight = np.matmul(np.linalg.inv(lamba*identity + np.matmul(np.transpose(class_attributes_seen),class_attributes_seen)),np.matmul(np.transpose(class_attributes_seen),mean))

for i in range(unseen_classes):
	mean_unseen[i] = np.matmul(class_attributes_unseen[i],weight)

correct_labels = 0

for i in range(input_number):
	min_dist = float("inf")
	min_index = 0
	
	for j in range(unseen_classes):
		temp = np.linalg.norm(mean_unseen[j]-Xtest[i])
		if(temp <= min_dist):
			min_dist = temp
			min_index = j+1

	Ypredicted[i] = min_index

	if(min_index == Ytest[i]):
		correct_labels += 1.0

#print accuracy
accuracy = correct_labels/input_number
print("Accuracy of Model for Lambda=1 is "+str(accuracy*100))

lamba = 10
weight = np.matmul(np.linalg.inv(lamba*identity + np.matmul(np.transpose(class_attributes_seen),class_attributes_seen)),np.matmul(np.transpose(class_attributes_seen),mean))

for i in range(unseen_classes):
	mean_unseen[i] = np.matmul(class_attributes_unseen[i],weight)

correct_labels = 0

for i in range(input_number):
	min_dist = float("inf")
	min_index = 0
	
	for j in range(unseen_classes):
		temp = np.linalg.norm(mean_unseen[j]-Xtest[i])
		if(temp <= min_dist):
			min_dist = temp
			min_index = j+1

	Ypredicted[i] = min_index

	if(min_index == Ytest[i]):
		correct_labels += 1.0

#print accuracy
accuracy = correct_labels/input_number
print("Accuracy of Model for Lambda=10 is "+str(accuracy*100))

lamba = 20
weight = np.matmul(np.linalg.inv(lamba*identity + np.matmul(np.transpose(class_attributes_seen),class_attributes_seen)),np.matmul(np.transpose(class_attributes_seen),mean))

for i in range(unseen_classes):
	mean_unseen[i] = np.matmul(class_attributes_unseen[i],weight)

correct_labels = 0

for i in range(input_number):
	min_dist = float("inf")
	min_index = 0
	
	for j in range(unseen_classes):
		temp = np.linalg.norm(mean_unseen[j]-Xtest[i])
		if(temp <= min_dist):
			min_dist = temp
			min_index = j+1

	Ypredicted[i] = min_index

	if(min_index == Ytest[i]):
		correct_labels += 1.0

#print accuracy
accuracy = correct_labels/input_number
print("Accuracy of Model for Lambda=20 is "+str(accuracy*100))

lamba = 50
weight = np.matmul(np.linalg.inv(lamba*identity + np.matmul(np.transpose(class_attributes_seen),class_attributes_seen)),np.matmul(np.transpose(class_attributes_seen),mean))

for i in range(unseen_classes):
	mean_unseen[i] = np.matmul(class_attributes_unseen[i],weight)

correct_labels = 0

for i in range(input_number):
	min_dist = float("inf")
	min_index = 0
	
	for j in range(unseen_classes):
		temp = np.linalg.norm(mean_unseen[j]-Xtest[i])
		if(temp <= min_dist):
			min_dist = temp
			min_index = j+1

	Ypredicted[i] = min_index

	if(min_index == Ytest[i]):
		correct_labels += 1.0

#print accuracy
accuracy = correct_labels/input_number
print("Accuracy of Model for Lambda=50 is "+str(accuracy*100))

lamba = 100
weight = np.matmul(np.linalg.inv(lamba*identity + np.matmul(np.transpose(class_attributes_seen),class_attributes_seen)),np.matmul(np.transpose(class_attributes_seen),mean))

for i in range(unseen_classes):
	mean_unseen[i] = np.matmul(class_attributes_unseen[i],weight)

correct_labels = 0

for i in range(input_number):
	min_dist = float("inf")
	min_index = 0
	
	for j in range(unseen_classes):
		temp = np.linalg.norm(mean_unseen[j]-Xtest[i])
		if(temp <= min_dist):
			min_dist = temp
			min_index = j+1

	Ypredicted[i] = min_index

	if(min_index == Ytest[i]):
		correct_labels += 1.0

#print accuracy
accuracy = correct_labels/input_number
print("Accuracy of Model for Lambda=100 is "+str(accuracy*100))