import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

diabetes_data = pd.read_csv('classification/data/diabetes_processed.csv')
print(diabetes_data.head())
print(diabetes_data.columns)

FEATURES = list(diabetes_data.columns[:-1])
print(FEATURES)


def apply_pca(n):
    pca = PCA(n_components=n)
    x_new = pca.fit_transform(diabetes_data[FEATURES])
    return pca, pd.DataFrame(x_new)


pca_obj, _ = apply_pca(8)

print('Explained Variance: ', pca_obj.explained_variance_ratio_)
print(sum(pca_obj.explained_variance_ratio_))

# visualization
# plt.figure(figsize=(8, 8))
# plt.plot(np.cumsum(pca_obj.explained_variance_ratio_))
# plt.xlabel('n components')
# plt.ylabel('cumulative variance')
# plt.show()

Y = diabetes_data['Outcome']
_, X_new = apply_pca(4)

x_train, x_test, y_train, y_test = train_test_split(X_new, Y, test_size=0.2)
model = LogisticRegression(solver='liblinear').fit(x_train, y_train)

y_pred = model.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))


