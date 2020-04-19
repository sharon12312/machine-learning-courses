import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA

wine_data = pd.read_csv('./data/winequality-white.csv',
                        names=['Fixed Acidity',
                               'Volatile Acidity',
                               'Citric Acid',
                               'Residual Sugar',
                               'Chlorides',
                               'Free Sulfur Dioxide',
                               'Total Sulfur Dioxide',
                               'Density',
                               'pH',
                               'Sulphates',
                               'Alcohol',
                               'Quality'],
                        skiprows=1,
                        sep=r'\s*;\s*',
                        engine='python')

print(wine_data.head())
print(wine_data['Quality'].unique())

X = wine_data.drop('Quality', axis=1)
Y = wine_data['Quality']

X = preprocessing.scale(X)

X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

clf_svc = LinearSVC(penalty='l1', dual=False, tol=1e-3)
clf_svc.fit(X_train, Y_train)

accuracy = clf_svc.score(x_test, y_test)
print(accuracy)

# Correlation visualization
# corrmat = wine_data.corr()
# f, ax = plt.subplots(figsize=(7, 7))
# sns.set(font_scale=0.9)
# sns.heatmap(corrmat, vmax=.8, square=True, annot=True, fmt='.2f', cmap='winter')
# plt.show()

# using PCA
# for 11 feature, for a start, we will use 11 components and then 9, 6, 1
pca = PCA(n_components=11, whiten=True)
X_reduced = pca.fit_transform(X)

# presents the dimensions, this way we can identify which one has a large rule on our data
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)

# visualization
# plt.plot(pca.explained_variance_ratio_)
# plt.xlabel('Dimension')
# plt.ylabel('Explain Variance Ratio')
# plt.show()

# training our model again using the PCA reduced features
X_train, x_test, Y_train, y_test = train_test_split(X_reduced, Y, test_size=0.2, random_state=0)
clf_svc_pca = LinearSVC(penalty='l1', dual=False, tol=1e-3)
clf_svc_pca.fit(X_train, Y_train)

# n_components=11 will change our accuracy, we need to reduce the number of features
accuracy = clf_svc_pca.score(x_test, y_test)
print(accuracy)
