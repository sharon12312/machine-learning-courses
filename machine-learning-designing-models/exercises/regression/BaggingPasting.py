import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import r2_score

insurance_data = pd.read_csv('data/insurance.csv')
print(insurance_data.head())
print(insurance_data.shape)

# correlation
insurance_data_correlation = insurance_data.corr()
print(insurance_data_correlation)

# visualization
# fig, ax = plt.subplots(figsize=(8, 8))
# sns.heatmap(insurance_data_correlation, annot=True)
# plt.show()

# pre processing
label_encoding = LabelEncoder()
insurance_data['region'] = label_encoding.fit_transform(insurance_data['region'].astype(str))
print(label_encoding.classes_)
insurance_data = pd.get_dummies(insurance_data, columns=['sex', 'smoker'])
print(insurance_data.sample(10))

# write to a file
insurance_data.to_csv('data/insurance_processed.csv', index=False)

# training a machine learning model
X = insurance_data.drop('charges', axis=1)
Y = insurance_data['charges']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# bagging - sample 80% of the samples in the dataset with replacement.
# oob=True will evaluate the ensembles on out-of-bag instances for each predictor and return a score
bag_reg = BaggingRegressor(DecisionTreeRegressor(), n_estimators=500, bootstrap=True, max_samples=0.8, n_jobs=-1, oob_score=True)
bag_reg.fit(x_train, y_train)
print(bag_reg.oob_score_)
y_pred = bag_reg.predict(x_test)
print(r2_score(y_test, y_pred))

# pasting - sample 90% of the samples in the dataset without replacement.
bag_reg = BaggingRegressor(DecisionTreeRegressor(), n_estimators=500, bootstrap=False, max_samples=0.9, n_jobs=-1)
bag_reg.fit(x_train, y_train)
y_pred = bag_reg.predict(x_test)
print(r2_score(y_test, y_pred))

