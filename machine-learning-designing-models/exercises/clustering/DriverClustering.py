import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

drivers_data = pd.read_csv('data/driver_dataset.csv', sep='\t')
print(drivers_data.head())
print(drivers_data.shape)

# shuffle the data
drivers_data = drivers_data.sample(frac=1)
drivers_data.drop('Driver_ID', axis=1, inplace=True)
print(drivers_data.sample(10))

# visualization
# fig, ax = plt.subplots(figsize=(10, 8))
# plt.scatter(drivers_data['Distance_Feature'], drivers_data['Speeding_Feature'], s=300, c='blue')
# plt.xlabel('Distance Feature')
# plt.ylabel('Speeding Feature')
# plt.show()

# clustering
kmeans_model = KMeans(n_clusters=4, max_iter=1000).fit(drivers_data)
print(kmeans_model.labels_[::40])
print(np.unique(kmeans_model.labels_))

# create a list which conains the data for features an labels
zipped_list = list(zip(np.array(drivers_data), kmeans_model.labels_))
print(zipped_list[1000:1010])  # an example

centroids = kmeans_model.cluster_centers_
print(centroids)

# visualization
# colors = ['g', 'y', 'b', 'k']
# plt.figure(figsize=(10, 8))
# for element in zipped_list:
#     plt.scatter(element[0][0], element[0][1], c=colors[(element[1] % len(colors))])
# plt.scatter(centroids[:,0], centroids[:,1], c='r', s=200, marker='s')
# for i in range(len(centroids)):
#     plt.annotate(i, (centroids[i][0], centroids[i][1]), fontsize=20)
# plt.show()

print('Silhouette score: ', silhouette_score(drivers_data, kmeans_model.labels_))

# n_clusters parameter was changes to 3
kmeans_model = KMeans(n_clusters=3, max_iter=1000).fit(drivers_data)
print(np.unique(kmeans_model.labels_))

zipped_list = list(zip(np.array(drivers_data), kmeans_model.labels_))
centroids = kmeans_model.cluster_centers_
print(centroids)

# colors = ['g', 'y', 'b', 'k']
# plt.figure(figsize=(10, 8))
# for element in zipped_list:
#     plt.scatter(element[0][0], element[0][1], c=colors[(element[1] % len(colors))])
# plt.scatter(centroids[:,0], centroids[:,1], c='r', s=200, marker='s')
# for i in range(len(centroids)):
#     plt.annotate(i, (centroids[i][0], centroids[i][1]), fontsize=20)
# plt.show()

print('Silhouette score: ', silhouette_score(drivers_data, kmeans_model.labels_))

# n_clusters parameter was changes to 2
kmeans_model = KMeans(n_clusters=2, max_iter=1000).fit(drivers_data)
print(np.unique(kmeans_model.labels_))

zipped_list = list(zip(np.array(drivers_data), kmeans_model.labels_))
centroids = kmeans_model.cluster_centers_
print(centroids)

# colors = ['g', 'y', 'b', 'k']
# plt.figure(figsize=(10, 8))
# for element in zipped_list:
#     plt.scatter(element[0][0], element[0][1], c=colors[(element[1] % len(colors))])
# plt.scatter(centroids[:,0], centroids[:,1], c='r', s=200, marker='s')
# for i in range(len(centroids)):
#     plt.annotate(i, (centroids[i][0], centroids[i][1]), fontsize=20)
# plt.show()

print('Silhouette score: ', silhouette_score(drivers_data, kmeans_model.labels_))
