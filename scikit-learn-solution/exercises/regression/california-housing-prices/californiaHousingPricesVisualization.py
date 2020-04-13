import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

housing_data = pd.read_csv('./data/housing.csv')
# print(housing_data.head())
# print(housing_data.sample(5))
# print(housing_data.shape)

# clean data set by performing dropna() function of pandas
housing_data = housing_data.dropna()
# print(housing_data.shape)
# print(housing_data.describe())
# print(housing_data['ocean_proximity'].unique())

fig, ax = plt.subplots(figsize=(12, 8))
plt.scatter(housing_data['total_rooms'], housing_data['median_house_value'])
plt.xlabel('Total rooms')
plt.ylabel('Median house value')
# plt.show()

fig, ax = plt.subplots(figsize=(12, 8))
plt.scatter(housing_data['housing_median_age'], housing_data['median_house_value'])
plt.xlabel('Median age')
plt.ylabel('Median house value')
# plt.show()

fig, ax = plt.subplots(figsize=(12, 8))
plt.scatter(housing_data['median_income'] * 10000, housing_data['median_house_value'])
plt.xlabel('Median income')
plt.ylabel('Median house value')
# plt.show()

housing_data_corr = housing_data.corr()
# print(housing_data_corr)

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(housing_data_corr, annot=True)
# plt.show(sns)
