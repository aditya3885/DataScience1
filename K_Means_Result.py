import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

##############################################################################
# Generate sample data
np.random.seed(0)

#batch_size = 45
centers = [[1, 1], [-1, -1], [1, -1]]
n_clusters = len(centers)
X, labels_true = make_blobs(n_samples=100, centers=centers, cluster_std=0.7)

##############################################################################
# Compute clustering with Means
#One iteration in K-Means 

kmeans = KMeans(n_clusters=3, random_state=0,max_iter=1).fit(X)

kmeans.predict([[0, 0], [2, 2],[6, 4]])

kmeans.cluster_centers_

############################
#Result for One iteration :
#array([[ 0.94956268,  0.66021792],
#       [-1.28644281, -0.47872578],
#       [ 0.55518985, -1.43879955]])

# Compute clustering with Means
k_means = KMeans(init='k-means++', n_clusters=3,max_iter=10).fit(X)

k_means_cluster_centers = k_means.cluster_centers_
##output of coordinates of centeroids at the end of 10th iteration 
#
#array([[ 0.99481806,  0.93803918],
#       [-1.15767412, -0.66201868],
#       [ 0.78248793, -1.13647079]])




k_means_labels = k_means.labels_


k_means_labels_unique = np.unique(k_means_labels)
#Output 2 set of unique cluster id that each point belong to :
# [0, 1, 2]



##############################################################################
# Plot result

colors = ['#4EACC5', '#FF9C34', '#4E9A06']
plt.figure()
#plt.hold(True)
for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
plt.title('KMeans')    
plt.grid(True)
plt.savefig("Kmeans"+".png",bbox_inches='tight')
plt.show()



###############################################################

# OUTPUT 

#Output 1 ----> set of unique cluster id that each point belong to :
# [0, 1, 2]




##Output 2-----> of coordinates of centeroids at the end of 10th iteration :

#array([[ 0.99481806,  0.93803918],
#       [-1.15767412, -0.66201868],
#       [ 0.78248793, -1.13647079]])