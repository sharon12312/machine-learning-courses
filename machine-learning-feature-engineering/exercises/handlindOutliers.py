import pandas as pd
import matplotlib.pyplot as plt

desired_width = 320
desired_columns = 5
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', desired_columns)

mall_customers_info = pd.read_csv('./data/Mall_Customers.csv')
print(mall_customers_info.head())

# print: (rows, columns)
print(mall_customers_info.shape)

# check if there are null values
print(mall_customers_info.isnull().any())

# get the number of unique values per column
print(mall_customers_info.nunique())

# plot the visualization of a specific column
# plt.show(mall_customers_info['Annual Income (k$)'].plot.hist(bins=10, figsize=(10, 8)))
# plt.show(mall_customers_info.plot.scatter(x='Age', y='Annual Income (k$)', c='DarkBlue', figsize=(10, 8)))
# plt.show(mall_customers_info.plot.scatter(x='Age', y='Annual Income (k$)', c='Spending Score (1-100)',
#                                         colormap='viridis', figsize=(10, 8)))

boxplot = mall_customers_info.boxplot(grid=False, rot=45, figsize=(10, 7))
# plt.show(boxplot)

boxplot = mall_customers_info.boxplot(grid=False, fontsize=15, column=['Annual Income (k$)'], figsize=(10, 8))
# plt.show(boxplot)

# find the outliers in a specific column
print(mall_customers_info.loc[mall_customers_info['Annual Income (k$)'] > 125])

Q1 = mall_customers_info['Annual Income (k$)'].quantile(0.25)
Q3 = mall_customers_info['Annual Income (k$)'].quantile(0.75)
IQR = Q3 - Q1

# initial a new column for finding an outlier
mall_customers_info['Annual_Income_Outlier'] = False

for index, row in mall_customers_info.iterrows():
    if row['Annual Income (k$)'] > (Q1 + 1.5 * IQR):  # a formula to detect outliers
        mall_customers_info.at[index, 'Annual_Income_Outlier'] = True

print(mall_customers_info['Annual_Income_Outlier'].sum())

non_outliers = mall_customers_info.loc[mall_customers_info['Annual_Income_Outlier'] == False]
print(non_outliers.head())

mean = non_outliers['Annual Income (k$)'].mean()

# covert the outlier to mean values
for index, row in mall_customers_info.iterrows():
    if row['Annual_Income_Outlier'] == True:
        mall_customers_info.at[index, 'Annual Income (k$)'] = mean

# see that there are no outliers anymore
boxplot = mall_customers_info.boxplot(grid=False, fontsize=15, column=['Annual Income (k$)'], figsize=(10, 8))
# plt.show(boxplot)
