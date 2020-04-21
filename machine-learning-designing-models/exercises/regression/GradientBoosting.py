import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

insurance_data = pd.read_csv('data/insurance_processed.csv')
print(insurance_data.head())

X = insurance_data.drop('charges', axis=1)
Y = insurance_data['charges']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# explanation regarding Gradient Boosting Regressor calculations
tree_reg1 = DecisionTreeRegressor(max_depth=3)
tree_reg1.fit(x_train, y_train)
y2 = y_train - tree_reg1.predict(x_train)

tree_reg2 = DecisionTreeRegressor(max_depth=3)
tree_reg2.fit(x_train, y2)
y3 = y2 - tree_reg2.predict(x_train)

tree_reg3 = DecisionTreeRegressor(max_depth=3)
tree_reg3.fit(x_train, y3)

y_pred = sum(tree.predict(x_test) for tree in (tree_reg1, tree_reg2, tree_reg3))
print(r2_score(y_test, y_pred))

# now the Gradient Boosting Regressor implementation.
# we will get the same result, because we used the same parameters as max_depth and n_estimators
gbr = GradientBoostingRegressor(max_depth=3, n_estimators=3, learning_rate=1.0)
gbr.fit(x_train, y_train)
y_pred = gbr.predict(x_test)
print(r2_score(y_test, y_pred))
