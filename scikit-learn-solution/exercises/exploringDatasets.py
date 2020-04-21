import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_digits

breast_cancer_dataset = load_breast_cancer()

# print(breast_cancer_dataset.keys())
# print(breast_cancer_dataset.DESCR)
# print(breast_cancer_dataset.feature_names)
# print(breast_cancer_dataset.data.shape)
# print(breast_cancer_dataset.target_names)
# print(breast_cancer_dataset.target.shape)

df_features = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
df_target = pd.DataFrame(breast_cancer_dataset.target, columns=['cancer'])
df = pd.concat([df_features, df_target], axis=1)
# print(df.head())
# print(df.shape)

# regression
boston_dataset = load_boston()

# classification - text data
fetch_20_train = fetch_20newsgroups(subset='train')

# classification - image data
digits_dataset = load_digits(n_class=10)

plt.imshow(digits_dataset.images[1], cmap='Greys')
# plt.show()