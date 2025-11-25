import matplotlib.pyplot as plt

def plot_clusters(X,labels,centroids):
    plt.scatter(X[:,0],X[:,1],c=labels)
    plt.scatter(centroids[:,0],centroids[:,1],marker='x',s=200)
    plt.title('Cluster Result')
    plt.savefig('clusters.png')