import pickle
import numpy as np 
import matplotlib.pyplot as plt 

with open('facedata.pkl', 'rb') as f:
    data = pickle.load(f)

data = np.asarray(data['X'])
num_dim = len(data[0])
num_images = len(data)

mean = []

for i in range(num_dim):
	count = 0
	num = 0
	for j in range(num_images):
		count += data[j][i]
		num+=1

	mean.append(count/num)
	for j in range(num_images):
		data[j][i] -= count/num

k=10
weight = np.zeros(shape=(num_dim,k))
latent = np.zeros(shape=(num_images,k))

for i in range(100):
	weight = np.matmul(np.matmul(np.transpose(data),latent),np.linalg.inv(np.matmul(np.transpose(latent),latent)))
	latent = np.matmul(np.matmul(data,weight),np.linalg.inv(np.matmul(np.transpose(weight),weight)))

for i in range(10):

	plt.imshow(np.reshape(weight[:,i],(64,64),order='F'))
	title_obj = plt.title("K="+str(k)+" W column image ") #get the title property handler
	plt.getp(title_obj)                    #print out the properties of title
	plt.getp(title_obj, 'text')            #print out the 'text' property for title
	plt.setp(title_obj, color='r')
	plt.savefig("k="+str(k)+"basis_image"+str(i+1)+".jpg")