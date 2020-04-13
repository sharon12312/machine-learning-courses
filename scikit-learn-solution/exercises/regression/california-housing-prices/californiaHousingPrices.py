import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

housing_data = pd.read_csv('./data/housing.csv')
housing_data = housing_data.dropna()

# check how many rows we have with the median house value of 50001
print(housing_data.loc[housing_data['median_house_value'] == 500001].count())

# we want to drop all the former values
housing_data = housing_data.drop(housing_data.loc[housing_data['median_house_value'] == 500001].index)

# get all the unique ocean_proximity values
print(housing_data['ocean_proximity'].unique())

# convert this values from string to columns using binary definition 1/0
# this method will enhance our table structure, each string value will become a column
housing_data = pd.get_dummies(housing_data, columns=['ocean_proximity'])
print(housing_data.sample(5))

X = housing_data.drop('median_house_value', axis=1)
Y = housing_data['median_house_value']

# split the data for 80% train and 20% test
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# train our model
linear_model = LinearRegression(normalize=True).fit(x_train, y_train)
print("Training score: ", linear_model.score(x_train, y_train))

# presents the coefficient of our features
# This way you can see how each coefficient lowers the house's price
# or raises the house's price (in ascending order).
predictors = x_train.columns
coef = pd.Series(linear_model.coef_, predictors).sort_values()
print(coef)

# Now we want to compare the forecast prices with the actual prices
y_pred = linear_model.predict(x_test)
df_pred_actual = pd.DataFrame({'predicted': y_pred, 'actual': y_test})
print(df_pred_actual.head(10))
print("Testing score: ", r2_score(y_test, y_pred))

fig, ax = plt.subplots(figsize=(12, 8))
plt.scatter(y_test, y_pred)
# plt.show()

df_pred_actual_sample = df_pred_actual.sample(100)
df_pred_actual_sample = df_pred_actual_sample.reset_index()
print(df_pred_actual_sample.head())

plt.figure(figsize=(20, 10))
plt.plot(df_pred_actual_sample['predicted'], label='Predicted')
plt.plot(df_pred_actual_sample['actual'], label='Actual')
plt.ylabel('median_house_values')
plt.legend()
# plt.show()

