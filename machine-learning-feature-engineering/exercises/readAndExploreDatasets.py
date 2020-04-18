import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

desired_width = 320
desired_columns = 10
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', desired_columns)

cars_data = pd.read_csv('./data/Cars93.csv')

# print data structure
print(cars_data.head())
print(cars_data.columns)
print(cars_data.shape)
print(cars_data.info())
print(cars_data.describe())
print(np.unique(cars_data['Manufacturer']))
print(cars_data[['Manufacturer', 'Model', 'MPG.city']].head())

cars_under_35 = cars_data[cars_data['Price'] <= 35]
print(cars_under_35.shape)

# locate by index
print(cars_data.iloc[[12, 19, 47]])

# investigate null values
print(cars_data.isnull().any(axis=0))
print(cars_data.isnull().sum())

cars_data_nulls = cars_data[cars_data.isnull().any(axis=1)]
print(cars_data_nulls[['Manufacturer', 'Model', 'Rear.seat.room', 'Luggage.room']])
print(cars_data_nulls.index)

cars_data = cars_data.dropna()
print(cars_data.shape)

# reduce the data columns
selected_columns = ['Manufacturer', 'Price', 'MPG.city', 'DriveTrain', 'EngineSize', 'Horsepower', 'Weight', 'Origin']
cars_data = cars_data[selected_columns]
print(cars_data.shape)
print(cars_data.head())

# visualize the data
plt.figure(figsize=(10, 8))
plt.scatter(cars_data['EngineSize'], cars_data['Horsepower'])
plt.title('Horsepower vs Engine Size')
plt.xlabel('Engine size (litres)')
plt.ylabel('Horsepower (bhp)')
# plt.show()

# categories string: transform the data from string to int.
#
# first approach - convert from string to int in ordinary convention ('A' -> 0, 'B' -> 1, etc.)
print(cars_data['Origin'].unique())
label_encoder = LabelEncoder()
cars_data['Origin'] = label_encoder.fit_transform(cars_data['Origin'])
print(cars_data.sample(5))

# second approach - one-hot encoding
# feature_1 | feature_2
#    A      |    xx
#    B      |    xx
#    C      |    xx
#    D      |    yy

# feature_2 | feature_1_A | feature_1_B | feature_1_C | feature_1_D
#    xx     |      1      |      1      |      1      |     0
#    yy     |      0      |      0      |      0      |     1
categorical_feature = cars_data.dtypes == object
print(categorical_feature)
print(cars_data['DriveTrain'].unique())
cars_data = pd.get_dummies(cars_data)
print(cars_data)
print(cars_data.shape)

# training one column model
X = cars_data[['Horsepower']]
Y = cars_data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

print(X_train.shape, y_train.size)
print(X_test.shape, y_test.shape)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
print(linear_model.score(X_train, y_train))

y_pred = linear_model.predict(X_test)
print(y_pred)

df = pd.DataFrame({'Test': y_test, 'Predicted': y_pred})
print(df.sample(10))
print(r2_score(y_test, y_pred))

# visualize the predictions
plt.figure(figsize=(10, 8))
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, c='r')
plt.title('Regression Line')
plt.xlabel('Horsepower (bhp)')
plt.ylabel('Price (1000s)')
# plt.show()

# training a simple model
X = cars_data.drop('Price', axis=1)
y = cars_data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
print(linear_model.score(X_train, y_train))
print(linear_model.score(X_test, y_test))

y_pred = linear_model.predict(X_test)
print(r2_score(y_test, y_pred))
