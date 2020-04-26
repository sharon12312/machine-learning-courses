import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv('data/123-processed-new.csv')
X = data.drop(['1', '2', '3', 'numbers'], axis=1)
Y = data[['numbers']]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

params = {'learning_rate': 0.4, 'max_depth': 20, 'n_estimators': 1000, 'loss': 'ls', 'random_state': 0,
          'min_samples_leaf': 2, 'min_samples_split': 3}

gbr_model = RandomForestRegressor(n_estimators=280, max_depth=28, min_samples_split=3, min_samples_leaf=2)
# gbr_model = GradientBoostingRegressor(**params)
gbr_model.fit(x_train, y_train)
print(gbr_model.score(x_train, y_train))

y_predict = gbr_model.predict(x_test)
print(r2_score(y_test, y_predict))

# visualization
plt.figure(figsize=(15, 6))
plt.plot(y_predict, label='Predicted')
plt.plot(y_test.values, label='Actual')
plt.legend()
# plt.show()

# {'learning_rate': 0.011, 'max_depth': 4, 'n_estimators': 135, 'loss': 'ls', 'random_state': 42}
# 0.5057476647612411
# 0.0012498090043662202
