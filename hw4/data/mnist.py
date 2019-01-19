import pickle as pk
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


# In[3]:


# Load data
file = open("mnist_small.pkl",'rb')
data = pk.load(file)
X = data['X']
Y = data['Y']


# In[4]:


# transform and fit
pca = PCA(n_components = 2)
X_trans = pca.fit(X).transform(X)

tsne = TSNE(n_components = 2,verbose=1)
X_emb = tsne.fit_transform(X)


# In[8]:


import matplotlib.pyplot as plt

color = ['blue','green','red','lightpink','purple','plum','aqua','olive','c','tan']

for i in range(10):
    # Random initialize means
    init_pca = X_trans[np.random.randint(low=0, high=X.shape[0]-1, size = (10))]
    init_tsne = X_emb[np.random.randint(low=0, high=X.shape[0]-1, size = (10))]
    
    # apply k-means on PCA embedding
    kmeans = KMeans(n_clusters = 10, init = init_pca).fit(X_trans)
    Y_pred_pca = kmeans.labels_
    col = np.where(Y_pred_pca == 0 ,color[0], np.where(Y_pred_pca == 1,color[1],        np.where(Y_pred_pca == 2,color[2],np.where(Y_pred_pca == 3,color[3],        np.where(Y_pred_pca == 4,color[4],np.where(Y_pred_pca == 5,color[5],        np.where(Y_pred_pca == 6 ,color[6], np.where(Y_pred_pca == 7,color[7],        np.where(Y_pred_pca == 8,color[8],color[9]                                        )))))))))
    title = "PCA-"+str(i)
    plt.title(title)
    plt.scatter(X_trans[:,0],X_trans[:,1],c = col)
    plt.scatter(init_pca[:,0],init_pca[:,1],marker='x')
    # plt.show()
    plt.savefig(title)
    
    print("Original")
    Y_pred_pca = Y.flatten()
    col = np.where(Y_pred_pca == 0 ,color[0], np.where(Y_pred_pca == 1,color[1],        np.where(Y_pred_pca == 2,color[2],np.where(Y_pred_pca == 3,color[3],        np.where(Y_pred_pca == 4,color[4],np.where(Y_pred_pca == 5,color[5],        np.where(Y_pred_pca == 6 ,color[6], np.where(Y_pred_pca == 7,color[7],        np.where(Y_pred_pca == 8,color[8],color[9]                                        )))))))))
    title = "PCA-"+str(i)
    plt.title(title)
    plt.scatter(X_trans[:,0],X_trans[:,1],c = col)
    plt.scatter(init_pca[:,0],init_pca[:,1],marker='x')
    # plt.show()
    
    # apply k-means on tsne embeddings
    kmeans = KMeans(n_clusters = 10, init = init_tsne).fit(X_emb)
    Y_pred_tsne = kmeans.labels_
    col = np.where(Y_pred_tsne == 0 ,color[0], np.where(Y_pred_tsne == 1,color[1],        np.where(Y_pred_tsne == 2,color[2],np.where(Y_pred_tsne == 3,color[3],        np.where(Y_pred_tsne == 4,color[4],np.where(Y_pred_tsne == 5,color[5],        np.where(Y_pred_tsne == 6 ,color[6], np.where(Y_pred_tsne == 7,color[7],        np.where(Y_pred_tsne == 8,color[8],color[9]                                        )))))))))
    title = "tsne-"+str(i)
    plt.title(title)
    plt.scatter(X_emb[:,0],X_emb[:,1],c = col)
    plt.scatter(init_tsne[:,0],init_tsne[:,1],marker='x')
    # plt.show()
    plt.savefig(title)