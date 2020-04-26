import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# data
advertising_data = pd.read_csv('data/Advertising.csv')
print(advertising_data.head())
print(advertising_data.shape)
print(advertising_data.describe())

# visualization
# plt.figure(figsize=(8, 8))
# plt.scatter(advertising_data['newspaper'], advertising_data['sales'], c='y')
# plt.show()
#
# plt.figure(figsize=(8, 8))
# plt.scatter(advertising_data['radio'], advertising_data['sales'], c='y')
# plt.show()
#
# plt.figure(figsize=(8, 8))
# plt.scatter(advertising_data['TV'], advertising_data['sales'], c='y')
# plt.show()

# correlation
advertising_data_correlation = advertising_data.corr()
print(advertising_data_correlation)

# fig, ax = plt.subplots(figsize=(8, 8))
# sns.heatmap(advertising_data_correlation, annot=True)
# plt.show()

# train and machine-learning-my-models our model
X = advertising_data['TV'].values.reshape(-1, 1)  # reshape to 2D array
Y = advertising_data['sales'].values.reshape(-1, 1)  # reshape to 2D array

print(X.shape, Y.shape)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
print(x_train.shape, y_train.shape)

# statistical models
x_train_with_intercept = sm.add_constant(x_train)
stats_model = sm.OLS(y_train, x_train_with_intercept)
fit_model = stats_model.fit()
print(fit_model.summary())

# liner model
linear_reg = LinearRegression(normalize=True).fit(x_train, y_train)
print('Training score: ', linear_reg.score(x_train, y_train))

y_pred = linear_reg.predict(x_test)
print('Test score: ', r2_score(y_test, y_pred))


# compute the adjusted r2 score
def adjusted_r2(r_square, labels, features):
    adj_r_square = 1 - ((1 - r_square) * (len(labels) - 1)) / (len(labels) - features.shape[1])
    return adj_r_square


print("Adjusted r2 score: ", adjusted_r2(r2_score(y_test, y_pred), y_test, x_test))

# plt.figure(figsize=(8, 8))
# plt.scatter(x_test, y_test, c='black')
# plt.scatter(x_test, y_pred, c='blue', linewidth=2)
# plt.xlabel('Money spent on TV ads ($)')
# plt.ylabel('Sales ($)')
# plt.show()

# ----------------------------

X = advertising_data.drop('sales', axis=1)
Y = advertising_data['sales']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

x_train_with_intercept = sm.add_constant(x_train)
stats_model = sm.OLS(y_train, x_train_with_intercept)
fit_model = stats_model.fit()
print(fit_model.summary())

linear_reg = LinearRegression(normalize=True).fit(x_train, y_train)
print('Training score: ', linear_reg.score(x_train, y_train))

y_pred = linear_reg.predict(x_test)
print('Test score: ', r2_score(y_test, y_pred))
print("Adjusted r2 score: ", adjusted_r2(r2_score(y_test, y_pred), y_test, x_test))
