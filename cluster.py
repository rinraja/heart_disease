#Unsupervised learning
import pandas as pd
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv", header=0)
#Print out the top of the table 
print(data.head(), data.shape)

#Try agglomerative clustering aka bottom up approach. Start with small clusters and build your way up
agg_dat = data.iloc[:,[0,3]].values

#Use a dendrogram in order to see the number of clusters that sshould be used
plt.figure(figsize=(13,10))
plt.title("Dendrogram of Heart Disease Data")
plt.xlabel("Clusters based on age and resting blood pressure")
plt.xticks(rotation=90)
plt.axhline(y=500, color='r', linestyle='--')
dendrogram_plot = shc.dendrogram(shc.linkage(agg_dat, method="ward"))

#Clustering #1
agg = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
agg.fit = agg.fit_predict(agg_dat)
#agg.fit(agg_dat)
labels = agg.labels_

#Plot the clusters in a scatter plot 
plt.figure(figsize=(10,7))
plt.scatter(agg_dat[labels==0,0],agg_dat[labels==0,1], color='orange', label='Cluster #1')
plt.scatter(agg_dat[labels==1,0],agg_dat[labels==1,1], color='blue', label='Cluster #2')
plt.scatter(agg_dat[labels==2,0],agg_dat[labels==2,1], color='red', label='Cluster #3')

plt.title("Clusters of Heart Disease Data")
plt.xlabel("Age of Patients")
plt.ylabel("Resting Blood Pressure")
plt.legend()

#Try agglomerative clustering aka bottom up approach. Start with small clusters and build your way up
agg_dat2 = data.iloc[:,[4,3]].values

#Use a dendrogram in order to see the number of clusters that sshould be used
plt.figure(figsize=(13,10))
plt.title("Dendrogram of Heart Disease Data")
plt.xlabel("Clusters based on resting blood pressure and cholesterol")
plt.xticks(rotation=90)
plt.axhline(y=500, color='r', linestyle='--')
dendrogram_plot = shc.dendrogram(shc.linkage(agg_dat2, method="ward"))

#Clustering #2 
agg2 = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
agg2.fit = agg2.fit_predict(agg_dat2)
labels = agg2.labels_

#Plot the clusters in a scatter plot 
plt.figure(figsize=(10,7))
plt.scatter(agg_dat2[labels==0,0],agg_dat2[labels==0,1], color='orange', label='Cluster #1')
plt.scatter(agg_dat2[labels==1,0],agg_dat2[labels==1,1], color='blue', label='Cluster #2')
plt.scatter(agg_dat2[labels==2,0],agg_dat2[labels==2,1], color='red', label='Cluster #3')

plt.title("Clusters of Heart Disease Data")
plt.xlabel("Cholesterol")
plt.ylabel("Resting Blood Pressure")
plt.legend()

plt.show()