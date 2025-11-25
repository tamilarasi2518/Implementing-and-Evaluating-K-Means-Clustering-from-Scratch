import numpy as np

def initialize_centroids(X,k):
    idx=np.random.choice(len(X),k,replace=False)
    return X[idx]

def assign_clusters(X,centroids):
    d=np.linalg.norm(X[:,None]-centroids[None,:],axis=2)
    return np.argmin(d,axis=1)

def update_centroids(X,labels,k):
    return np.array([X[labels==i].mean(axis=0) for i in range(k)])

def kmeans(X,k,iters=100):
    centroids=initialize_centroids(X,k)
    for _ in range(iters):
        labels=assign_clusters(X,centroids)
        new_centroids=update_centroids(X,labels,k)
        if np.allclose(new_centroids,centroids):
            break
        centroids=new_centroids
    return labels,centroids

def inertia(X,labels,centroids):
    return sum(np.linalg.norm(X[labels==i]-centroids[i])**2 for i in range(len(centroids)))