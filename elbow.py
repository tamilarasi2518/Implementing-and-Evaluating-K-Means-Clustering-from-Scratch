import numpy as np
import matplotlib.pyplot as plt
from kmeans import kmeans,inertia

def elbow_method(X,max_k=10):
    inertias=[]
    ks=range(1,max_k+1)
    for k in ks:
        labels,centroids=kmeans(X,k)
        inertias.append(inertia(X,labels,centroids))
    plt.plot(ks,inertias,marker='o')
    plt.title('Elbow Method')
    plt.xlabel('K')
    plt.ylabel('Inertia')
    plt.savefig('elbow.png')