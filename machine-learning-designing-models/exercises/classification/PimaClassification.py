import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

diabetes_data = pd.read_csv('data/diabetes.csv')
print(diabetes_data.head())
print(diabetes_data.columns)
print(diabetes_data.shape)
print(diabetes_data.describe())

# pre processing
# if the outcome was string, we'll encode it to numeric format
# label_encoding = preprocessing.LabelEncoder()
# diabetes_data['Outcome'] = label_encoding.fit_transform(diabetes_data['Outcome'].astype(str))
# print(diabetes_data.sample(10))
# print(label_encoding.classes_)

# visualization
# plt.figure(figsize=(8, 8))
# plt.scatter(diabetes_data['Glucose'], diabetes_data['Outcome'], c='g')
# plt.xlabel('Glucose')
# plt.ylabel('Outcome')
# plt.show()

# plt.figure(figsize=(8, 8))
# plt.scatter(diabetes_data['Age'], diabetes_data['Insulin'], c='g')
# plt.xlabel('Age')
# plt.ylabel('Insulin')
# plt.show()

# correlation
diabetes_data_correlation = diabetes_data.corr()
print(diabetes_data_correlation)

# fig, ax = plt.subplots(figsize=(8, 8))
# sns.heatmap(diabetes_data_correlation, annot=True)
# plt.show()

features = diabetes_data.drop('Outcome', axis=1)
labels = diabetes_data['Outcome']

# standard deviation
standard_scaler = preprocessing.StandardScaler()
features_scaled = standard_scaler.fit_transform(features)
features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
print(features_scaled_df.head())
print(features_scaled_df.describe())

diabetes_data = pd.concat([features_scaled_df, diabetes_data['Outcome']], axis=1).reset_index(drop=True)
diabetes_data.to_csv('data/diabetes_processed.csv', index=False)

# train and test our model: LogisticRegression
X = diabetes_data.drop('Outcome', axis=1)
Y = diabetes_data['Outcome']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
classifier = LogisticRegression(penalty='l2', C=1.0, solver='liblinear')
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
pred_results = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
print(pred_results.head(5))

# scores
model_accuracy = accuracy_score(y_test, y_pred)
model_precision = precision_score(y_test, y_pred)
model_recall = recall_score(y_test, y_pred)

print('Logistic Regression Classifier:')
print('Accuracy of the model is {:.2f}%'.format(model_accuracy * 100))
print('Precision of the model is {:.2f}%'.format(model_precision * 100))
print('Recall of the model is {:.2f}%'.format(model_recall * 100))

# train and test our model: DecisionTreeClassifier
classifier = DecisionTreeClassifier(max_depth=4)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
pred_results = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})

model_accuracy = accuracy_score(y_test, y_pred)
model_precision = precision_score(y_test, y_pred)
model_recall = recall_score(y_test, y_pred)

print('Decision Tree Classifier:')
print('Accuracy of the model is {:.2f}%'.format(model_accuracy * 100))
print('Precision of the model is {:.2f}%'.format(model_precision * 100))
print('Recall of the model is {:.2f}%'.format(model_recall * 100))

# confusion matrix
diabetes_crosstab = pd.crosstab(pred_results.y_pred, pred_results.y_test)
print(diabetes_crosstab)

TP = diabetes_crosstab[1][1]
TN = diabetes_crosstab[0][0]
FP = diabetes_crosstab[0][1]
FN = diabetes_crosstab[1][0]

accuracy_score_verified = (TP + TN) / (TP + FP + TN + FN)
print(accuracy_score_verified)

precision_score_survived = TP / (TP + FP)
print(precision_score_survived)

recall_score_survived = TP / (TP + FN)
print(recall_score_survived)
