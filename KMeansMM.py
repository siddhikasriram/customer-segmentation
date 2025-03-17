# -*- coding: utf-8 -*-
"""
INFSY 566- Kmeans Clustering, dropping colums from dataset and Standard Silhouette Graph
Min Max Scaling of data before cluster analysis

"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# select a file from data directory
df=pd.read_csv('simClass.csv', header=None) # use it when file does not have headers

# make it 2-D by extracting first and second columns
X_ = df.iloc[0:100,[0,1]].values 

from sklearn.preprocessing import MinMaxScaler
mms=MinMaxScaler()
X=mms.fit_transform(X_)

from sklearn.cluster import KMeans

# n_init=10 means run 10 experiments and choose the best model based on SSE
km = KMeans(n_clusters=2, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
y_km = km.fit_predict(X) # predict cluster labels

# Print Cluster Centroids (useful for Homework)
print('Cluster 1 Centroid:', km.cluster_centers_[0, 0], km.cluster_centers_[0, 1])
print('Cluster 2 Centroid:', km.cluster_centers_[1, 0], km.cluster_centers_[1, 1])


#print(y_km) # print cluster labels for each example

# This is just a scatter plot of points in unicolor

#plt.scatter(X[:100,0], X[:100,1], c='white', marker='o', edgecolor='black', s=50)

# following plots should be all run together to generate final output. Running individually can be messy

# for all data where y_km is zero is first cluster X[0] and X[1] where y_km=0
plt.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            s=50, c='lightgreen',
            marker='s', edgecolor='black',
            label='cluster 1')
#for all data where y_km is 1 is second cluster
plt.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            s=50, c='orange',
            marker='o', edgecolor='black',
            label='cluster 2')

plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250, marker='*',
            c='red', edgecolor='black',
            label='centroids')
plt.legend(scatterpoints=1)
plt.grid()
plt.tight_layout()
plt.savefig('cluster.png', dpi=300) # This will generate a image to use in your final reports
plt.show()




# Generate silhouette graph...  Standard code that uses many methods and looks at many clusters
# on the datasets beginning with 1 to the max clusters we used in our analysis above (2 in our case)
# This will be useful when we have data containing more than 2-D where we cannot draw a graph to visually
# inspect things the way we do in our 2D dataset

from sklearn.metrics import silhouette_samples
from matplotlib import cm
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, 
             edgecolor='none', color=color)

    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)
    
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--") 

plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')

plt.tight_layout()
#plt.savefig('Silhouette.png', dpi=300)
plt.show()


 






