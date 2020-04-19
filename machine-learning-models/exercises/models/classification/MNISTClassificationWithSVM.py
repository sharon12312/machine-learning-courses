import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

mnist_data = pd.read_csv('../data/mnist_train.csv')
print(mnist_data.tail())

features = mnist_data.columns[1:]
X = mnist_data[features]
Y = mnist_data['label']

X_train, x_test, Y_train, y_test = train_test_split(X/255., Y, test_size=0.1, random_state=0)

clf_svm = LinearSVC(penalty='l2', dual=False, tol=1e-5)
clf_svm.fit(X_train, Y_train)

y_pred_svm = clf_svm.predict(x_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy: ", acc_svm)

# grid search help us to fetch the best parameters for our model
penalties = ['l1', 'l2']
tolerances = [1e-3, 1e-4, 1e-5]
param_grid = {'penalty': penalties, 'tol': tolerances}

# # the grid search can take some time, because it will check all the permutations
# # to get the best model
# grid_search = GridSearchCV(LinearSVC(dual=False), param_grid, cv=3)
# grid_search.fit(X_train, Y_train)
# print(grid_search.best_params_)  # expected result: {'penalty': 'l1', 'tol': 0.0001/1e-4}

# according to our result:
clf_svm = LinearSVC(penalty='l1', dual=False, tol=1e-4)
clf_svm.fit(X_train, Y_train)

y_pred_svm = clf_svm.predict(x_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy: ", acc_svm)
