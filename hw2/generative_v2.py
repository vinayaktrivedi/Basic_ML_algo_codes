import csv
import numpy as np 
import matplotlib.pyplot as plt
import math
from sklearn import svm 


#open the data file and calculate number of examples
data = open("binclassv2.txt", "r")
csvReader = csv.reader(data, delimiter=',')
num_examples = 0
for row in csvReader:
	num_examples+=1

#open the data file and extract the feature vectors and classes separately in variables
data = open("binclassv2.txt", "r")
csvReader = csv.reader(data, delimiter=',')
num_features = 2
data_tab = np.zeros(shape=(num_examples,num_features),dtype=float)
classes = []
j=0
for row in csvReader:
	for i in range(num_features):
		data_tab[j][i] = row[i]
	classes.append(row[i+1])
	j+=1


#number of examples from each class
numclass_0=0
numclass_1=0


#calculate mean of each class by formula
mean_0=np.zeros(shape=(1,num_features),dtype=float)
mean_1=np.zeros(shape=(1,num_features),dtype=float)
for i in range(num_examples):
	
	if(classes[i]=="-1"):
		numclass_0+=1
		mean_0+=data_tab[i]
	else:
		numclass_1+=1
		mean_1+=data_tab[i]

mean_0 = mean_0/numclass_0
mean_1 = mean_1/numclass_1


#calculate variance of each class where variance are assumed to be different
#plot_x and plot_y have respective first and second feature values in order 
#-  to make the scatter plot where we pass respective feature arrays and classes


var_0=0
var_1=0
plot_x_0=[]
plot_y_0=[]
plot_x_1=[]
plot_y_1=[]
for i in range(num_examples):
	temp = np.zeros(shape=(1,num_features),dtype=float)
	if(classes[i] == "-1"):
		temp = data_tab[i] - mean_0
		var_0 += np.inner(temp,temp)
		plot_x_0.append(data_tab[i][0])
		plot_y_0.append(data_tab[i][1])


	else:
		temp = data_tab[i] - mean_1
		var_1 += np.inner(temp,temp)
		plot_x_1.append(data_tab[i][0])
		plot_y_1.append(data_tab[i][1])
	

var_0 = var_0/(num_features*numclass_0)
var_1 = var_1/(num_features*numclass_1)


#Plot the scatter plot containing the respective classes
plt.scatter(plot_x_0, plot_y_0, c="red",label="class -1")
plt.scatter(plot_x_1, plot_y_1, c="blue",label="class 1")

#plot the respective non-linear decision boundary and save the fig
x_range = np.linspace(-12, 45)   
y_range = np.linspace(-12, 45)[:, None] 
xc, yc = np.meshgrid(x_range, y_range)

plt.contour(x_range, y_range.ravel(), ((1/(2*var_0))*((xc-mean_0[0][0])**2+(yc-mean_0[0][1])**2))-((1/(2*var_1))*((xc-mean_1[0][0])**2+(yc-mean_1[0][1])**2))+(math.log(var_0)-math.log(var_1)), [0])
plt.title('Second data file, Generative with different variance')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc=2)
plt.savefig("v2_non")
plt.close()


#plot the classes for second case where variance is same
plt.scatter(plot_x_0, plot_y_0, c="red",label="class -1")
plt.scatter(plot_x_1, plot_y_1, c="blue",label="class 1")


#make variance same, plot the linear decision boundary and save the figure
var_0=var_1
plt.contour(x_range, y_range.ravel(), ((1/(2*var_0))*((xc-mean_0[0][0])**2+(yc-mean_0[0][1])**2))-((1/(2*var_1))*((xc-mean_1[0][0])**2+(yc-mean_1[0][1])**2))+(math.log(var_0)-math.log(var_1)), [0])
plt.title('Data file 2, Generative with equal variance')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc=2)
plt.savefig("v2_lin")
plt.close()

#Linear SVM case, with C=1
clf = svm.SVC(kernel="linear")
clf.fit(data_tab,classes)

w = clf.coef_
b = clf.intercept_


#plot the classes for linear SVM case
plt.scatter(plot_x_0, plot_y_0, c="red",label="class -1")
plt.scatter(plot_x_1, plot_y_1, c="blue",label="class 1")


#plot the linear boundary in SVM case and save the figure
plt.contour(x_range,y_range.ravel(),xc*w[0][0]+yc*w[0][1]+b,[0])
print(w)
print(b)
plt.title('Linear SVM for datafile 2')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc=2)
plt.savefig("v2_svm")
plt.close()
