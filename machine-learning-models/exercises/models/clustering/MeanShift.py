import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth

titanic_data = pd.read_csv('../../data/train.csv', quotechar='"')
print(titanic_data.head())

# drop features which are too specific to individual passengers
titanic_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], 'columns', inplace=True)
print(titanic_data.head())

le = preprocessing.LabelEncoder()
titanic_data['Sex'] = le.fit_transform(titanic_data['Sex'].astype(str))
print(titanic_data.head())

titanic_data = pd.get_dummies(titanic_data, columns=['Embarked'])
print(titanic_data.head())

# check for null values
print(titanic_data[titanic_data.isnull().any(axis=1)])
titanic_data = titanic_data.dropna()

# perform clustering
# starting from 50 and then change it to 30 according to the estimated_bandwidth() function
analyzer = MeanShift(bandwidth=50)
analyzer.fit(titanic_data)

# helper function to help estimate a good value fro bandwidth
print(estimate_bandwidth(titanic_data))

labels = analyzer.labels_
print(np.unique(labels))

# include the cluster in the same data frame
titanic_data['cluster_group'] = np.nan
data_length = len(titanic_data)

for i in range(data_length):
    titanic_data.iloc[i, titanic_data.columns.get_loc('cluster_group')] = labels[i]

print(titanic_data.sample(10))
print(titanic_data.describe())

# group by cluster and present the average for each cluster group
titanic_cluster_data = titanic_data.groupby(['cluster_group']).mean()
print(titanic_cluster_data)

titanic_cluster_data['Counts'] = pd.Series(titanic_data.groupby(['cluster_group']).size())
print(titanic_cluster_data)

# let's deep into one of the classes
print(titanic_data[titanic_data['cluster_group'] == 1].describe())
print(titanic_data[titanic_data['cluster_group'] == 1])
