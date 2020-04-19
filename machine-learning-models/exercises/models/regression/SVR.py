import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from pandas import Series
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

auto_data = pd.read_csv('../data/auto-mpg.data', delim_whitespace=True, header=None,
                        names=['mpg','cylinders', 'displacement', 'horsepower', 'weight','acceleration', 'model',
                               'origin', 'car_name'])
print(auto_data.head())

# the car name will now effect our training
print(len(auto_data['car_name'].unique()))
print(len(auto_data))

auto_data = auto_data.drop('car_name', axis=1)
print(auto_data.head())

# convert origin to one-hot presentation
print(auto_data['origin'].unique())
origin_dict = {1: 'america', 2: 'europe', 3: 'asia'}
auto_data['origin'] = auto_data['origin'].replace(origin_dict)
auto_data = pd.get_dummies(auto_data, columns=['origin'])
print(auto_data.sample(5))

# handling missing values
auto_data = auto_data.replace('?', np.nan)
auto_data = auto_data.dropna()

# training our model
X = auto_data.drop('mpg', axis=1)
Y = auto_data['mpg']

X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# start with C=1.0 to 0.5 (better model)
regression_model = SVR(kernel='linear', C=0.5)
regression_model.fit(X_train, Y_train)

print(regression_model.score(X_train, Y_train))

predictors = X_train.columns
coef = Series(regression_model.coef_[0], predictors).sort_values()
# coef.plot(kind='bar', title='Modal Coefficients')
# plt.show()

y_predict = regression_model.predict(x_test)

# plt.figure(figsize=(15, 6))
# plt.plot(y_predict, label='Predicted')
# plt.plot(y_test.values, label='Actual')
# plt.ylabel('MGP')
# plt.legend()
# plt.show()

print(regression_model.score(x_test, y_test))

regression_model_mse = mean_squared_error(y_predict, y_test)
print(regression_model_mse)
print(math.sqrt(regression_model_mse))
