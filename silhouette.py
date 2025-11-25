import numpy as np
from kmeans import kmeans
from sklearn.metrics import silhouette_score

def silhouette_analysis(X,k):
    labels,centroids=kmeans(X,k)
    return silhouette_score(X,labels)