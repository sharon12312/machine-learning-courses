import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

auto_data = pd.read_csv('../data/imports-85.data', sep=r'\s*,\s*', engine='python')
print(auto_data)

auto_data = auto_data.replace('?', np.nan)
print(auto_data.head())

print(auto_data.describe(include='all'))
print(auto_data['price'].describe())

# convert object price to float
auto_data['price'] = pd.to_numeric(auto_data['price'], errors='coerce')
print(auto_data['price'].describe())

# drop unrelevant columns
auto_data = auto_data.drop('normalized-losses', axis=1)
print(auto_data.head())

# convert other objects to float/one-hot representation
auto_data['horsepower'] = pd.to_numeric(auto_data['horsepower'], errors='coerce')
print(auto_data['horsepower'].describe())

cylinders_dict = {'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'eight': 8, 'twelve': 12}
auto_data['num-of-cylinders'].replace(cylinders_dict, inplace=True)

auto_data = pd.get_dummies(auto_data, columns=['make', 'fuel-type', 'aspiration', 'num-of-doors',
                                               'body-style', 'drive-wheels', 'engine-location',
                                               'engine-type', 'fuel-system'])

print(auto_data.columns)
print(auto_data.shape)

# drop all rows with null values
auto_data = auto_data.dropna()
print(auto_data.shape)

# check if there are null values
print(auto_data.isnull().sum())
print(auto_data[auto_data.isnull().any(axis=1)])

# training our regression model
X = auto_data.drop('price', axis=1)
Y = auto_data['price']

X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

linear_model = LinearRegression()
linear_model.fit(X_train, Y_train)
print(linear_model.score(X_train, Y_train))

# coefficiency presents all the points on the graph
# the smallest numbers will lower the prices while the highest number will increase the prices
# 0 values will not effect our prices
predictors = X_train.columns
coef = pd.Series(linear_model.coef_, predictors).sort_values()
print(coef)

y_predict = linear_model.predict(x_test)

# visualization
# plt.figure(figsize=(15, 6))
# plt.plot(y_predict, label='Predicted')
# plt.plot(y_test.values, label='Actual')
# plt.ylabel('Price')
# plt.legend()
# plt.show()

r_square = linear_model.score(x_test, y_test)
print(r_square)

liner_model_mse = mean_squared_error(y_predict, y_test)
print(liner_model_mse)

# present the error
# in our example the output will be the average amount of the price which is away form our actual price
# in a positive ot negative direction
print(math.sqrt(liner_model_mse))

# ---------------------

# start with alpha=0.5 then change it to alpha=5 to see which one is better for our model
lasso_model = Lasso(alpha=0.5, normalize=True)
lasso_model.fit(X_train, Y_train)

print(lasso_model.score(X_train, Y_train))
coef = pd.Series(lasso_model.coef_, predictors).sort_values()
print(coef)

y_predict = lasso_model.predict(x_test)

# plt.figure(figsize=(15, 6))
# plt.plot(y_predict, label='Predicted')
# plt.plot(y_test.values, label='Actual')
# plt.ylabel('Price')
# plt.legend()
# plt.show()

r_square = lasso_model.score(x_test, y_test)
print(r_square)

lasso_model_mse = mean_squared_error(y_predict, y_test)
print(math.sqrt(lasso_model_mse))

# ---------------------

# start with alpha=0.05, 0.5 and then 1.0 => 0.5 is the best for our model
ridge_model = Ridge(alpha=1.0, normalize=True)
ridge_model.fit(X_train, Y_train)

print(ridge_model.score(X_train, Y_train))
coef = pd.Series(ridge_model.coef_, predictors).sort_values()
print(coef)

y_predict = ridge_model.predict(x_test)

# plt.figure(figsize=(15, 6))
# plt.plot(y_predict, label='Predicted')
# plt.plot(y_test.values, label='Actual')
# plt.ylabel('Price')
# plt.legend()
# plt.show()

r_square = ridge_model.score(x_test, y_test)
print(r_square)

ridge_model_mse = mean_squared_error(y_predict, y_test)
print(math.sqrt(ridge_model_mse))
