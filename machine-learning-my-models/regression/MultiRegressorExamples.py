import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def print_result(title, y_test, y_pred):
    print('{} Results:'.format(title))
    print('---actual---')
    for item in y_test.values:
        numbers = list(item)
        print('{0},{1},{2}'.format(numbers[0], numbers[1], numbers[2]))
    print('---predicted---')
    for item in y_pred:
        numbers = list(item)
        print('{0},{1},{2}'.format(math.floor(numbers[0]), math.floor(numbers[1]), math.floor(numbers[2])))
    print('')


# dataset
data = pd.read_csv('data/123-processed.csv')
X = data.drop(['1', '2', '3'], axis=1)
Y = data[['1', '2', '3']]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# reg_rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=400, max_depth=20, random_state=0))

params = {'n_estimators': 10, 'max_depth': 6, 'min_samples_split': 2, 'learning_rate': 0.5, 'loss': 'ls'}
gbr_model = GradientBoostingRegressor(**params)

regr_multirf = MultiOutputRegressor(gbr_model)
regr_multirf.fit(x_train, y_train)
print('Multi Output Regressor score:', regr_multirf.score(x_train, y_train))

regr_rf = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=0)
regr_rf.fit(x_train, y_train)
print('Random Forest Regressor score', regr_rf.score(x_train, y_train))
print('')

# predict on new data
y_multirf = regr_multirf.predict(x_test)
# print_result('Multi Output Regressor', y_test, y_multirf)

y_rf = regr_rf.predict(x_test)
# print_result('Random Forest Regressor', y_test, y_rf)

# visualization
plt.figure(figsize=(10, 8))
s = 50
a = 0.4
plt.scatter(y_test.iloc[:, 0], y_test.iloc[:, 1], edgecolor='k', c='navy', s=s, marker='s', alpha=a, label='Data')
plt.scatter(y_multirf[:, 0], y_multirf[:, 1], edgecolor='k', c='cornflowerblue', s=s,
            alpha=a, label='Multi RF score=%.2f' % regr_multirf.score(x_test, y_test))
plt.scatter(y_rf[:, 0], y_rf[:, 1], edgecolor='k', c='c', s=s, marker="^", alpha=a,
            label='RF score=%.2f' % regr_rf.score(x_test, y_test))
plt.xlim([-1, 10])
plt.ylim([-1, 10])
plt.xlabel('target 1')
plt.ylabel('target 2')
plt.title('Comparing random forests and the multi-output meta estimator')
plt.legend()
plt.show()
