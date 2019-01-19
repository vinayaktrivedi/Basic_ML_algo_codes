import numpy as np  

#load X_seen
X_seen = np.load('X_seen.npy')

#number of seen classes
seen_classes = len(X_seen)

#number of features
features = X_seen[0].shape[1]

#mean of seen classes
mean = np.zeros(shape=(seen_classes,features),dtype=float)


#Compute mean of each seen class
for i in range(seen_classes):	

	for j in range(X_seen[i].shape[0]):
		mean[i] += X_seen[i][j]
	mean[i] = mean[i]/X_seen[i].shape[0]

#load attributes of seen and unseen class
class_attributes_seen = np.load('class_attributes_seen.npy')
class_attributes_unseen = np.load('class_attributes_unseen.npy')

#number of unseen classes
unseen_classes = len(class_attributes_unseen)

#similarity matrix of unseen classes wrt seen
dot_product = np.zeros(shape=(unseen_classes,seen_classes),dtype=float)

#compute similarity matrix and normalize it
for i in range(unseen_classes):
	for j in range(seen_classes):
		dot_product[i][j] = np.dot(class_attributes_unseen[i],class_attributes_seen[j])

	dot_product[i] = dot_product[i]/np.sum(dot_product[i])


#compute of mean of unseen classes
mean_unseen = np.zeros(shape=(unseen_classes,features),dtype=float)
for i in range(unseen_classes):
	for j in range(seen_classes):
		mean_unseen[i] += dot_product[i][j]*mean[j]


#load test data
Xtest = np.load('Xtest.npy')
Ytest = np.load('Ytest.npy')

#number of test inputs
input_number = len(Xtest)

Ypredicted = np.ndarray(shape=(input_number,1))
correct_labels = 0

#prediction on test data and calculate accuracy
for i in range(input_number):
	min_dist = float("inf")
	min_index = 0
	
	for j in range(unseen_classes):
		#euclidean distance between mean of classes and test inputs
		temp = np.linalg.norm(mean_unseen[j]-Xtest[i])
		if(temp < min_dist):
			min_dist = temp
			min_index = j+1

	Ypredicted[i] = min_index

	if(min_index == Ytest[i]):
		correct_labels += 1.0

#print accuracy
accuracy = correct_labels/input_number
print("Accuracy of Convex Model is "+str(accuracy*100)) 








	






