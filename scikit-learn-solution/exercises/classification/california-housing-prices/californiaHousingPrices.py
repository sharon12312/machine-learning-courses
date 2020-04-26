import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# pre-processing our dataset
housing_data = pd.read_csv('./data/housing.csv')
housing_data = housing_data.dropna()
housing_data = housing_data.drop(housing_data.loc[housing_data['median_house_value'] == 500001].index)
housing_data = pd.get_dummies(housing_data, columns=['ocean_proximity'])

# get the median, this way we can perform a classification in our regression model
median = housing_data['median_house_value'].median()
print("Median: ", median)

# this way we can create our labels. 'above_median' will set to True/False
housing_data['above_median'] = (housing_data['median_house_value'] - median) > 0

X = housing_data.drop(['median_house_value', 'above_median'], axis=1)
Y = housing_data['above_median']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# using 'liblinear as a solver is good for small dataset and binary classification
logistic_model = LogisticRegression(solver='liblinear').fit(x_train, y_train)
print("Training score: ", logistic_model.score(x_train, y_train))

# get the score of our machine-learning-my-models data
y_pred = logistic_model.predict(x_test)
df_pred_actual = pd.DataFrame({'predicted': y_pred, 'actual': y_test})
print(df_pred_actual.head(10))
print("Test score: ", accuracy_score(y_test, y_pred))
