import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
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

params = {'n_estimators': 500, 'max_depth': 6, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}

gbr_model = GradientBoostingRegressor(**params)
gbr_model.fit(X_train, Y_train)

print(gbr_model.score(X_train, Y_train))

y_predict = gbr_model.predict(x_test)

# visualization
# plt.figure(figsize=(15, 6))
# plt.plot(y_predict, label='Predicted')
# plt.plot(y_test.values, label='Actual')
# plt.ylabel('MPG')
# plt.legend()
# plt.show()

r_square = gbr_model.score(x_test, y_test)
print(r_square)

gbr_model_mse = mean_squared_error(y_test, y_predict)
print(gbr_model_mse)
print(math.sqrt(gbr_model_mse))

# Grid Search CV
num_estimators = [100, 200, 500]
learn_rates = [0.01, 0.02, 0.05, 0.1]
max_depths = [4, 6, 8]

param_grid = {
    'n_estimators': num_estimators,
    'learning_rate': learn_rates,
    'max_depth': max_depths
}

grid_search = GridSearchCV(GradientBoostingRegressor(min_samples_split=2, loss='ls'),
                           param_grid, cv=3, return_train_score=True)
grid_search.fit(X_train, Y_train)
print(grid_search.best_params_)
# print(grid_search.cv_results_)

# result: {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.05}

for i in range(36):  # 3 estimators * 4 learning rates * 3 max depths = 36
    print('Parameters: ', grid_search.cv_results_['params'][i])
    print('Mean Test Score: ', grid_search.cv_results_['mean_test_score'][i])
    print('Rank: ', grid_search.cv_results_['rank_test_score'][i])
    print()

# training our model with the expected parameters from the grid search
params = {'n_estimators': 200, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.05, 'loss': 'ls'}

gbr_model = GradientBoostingRegressor(**params)
gbr_model.fit(X_train, Y_train)

print(gbr_model.score(X_train, Y_train))

y_predict = gbr_model.predict(x_test)

# # visualization
# plt.figure(figsize=(15, 6))
# plt.plot(y_predict, label='Predicted')
# plt.plot(y_test.values, label='Actual')
# plt.ylabel('MPG')
# plt.legend()
# plt.show()

r_square = gbr_model.score(x_test, y_test)
print(r_square)

gbr_model_mse = mean_squared_error(y_test, y_predict)
print(gbr_model_mse)
print(math.sqrt(gbr_model_mse))
