import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from yellowbrick.target import FeatureCorrelation

melb_data = pd.read_csv('../data/melb_data_processed.csv')

print(melb_data.describe().transpose().round(2))
print(melb_data.isnull().sum())

py_num_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_columns = list(melb_data.select_dtypes(include=py_num_types).columns)
print(numerical_columns)

# set only numerical columns
clean_data_numeric = melb_data[numerical_columns]
print(clean_data_numeric.sample(10))

# drop unrelevant columns
clean_data_numeric = clean_data_numeric.drop(['Lattitude', 'Longtitude', 'Postcode'], axis=1)
print(clean_data_numeric.shape)
print(clean_data_numeric.info())

# correlation
pearson_corr = clean_data_numeric.corr(method='pearson')  # linear correlation
print(pearson_corr)

spearman_corr = clean_data_numeric.corr(method='spearman')  # ordered non linear correlation
print(spearman_corr)

kendall_corr = clean_data_numeric.corr(method='kendall')  # unordered non linear correlation
print(kendall_corr)

# visualization
# plt.figure(figsize=(10, 8))
# sns.heatmap(pearson_corr, linewidth=1, annot=True, annot_kws={'size': 10})
# plt.title('Pearson Correlation', fontsize=25)
# plt.show()

# plt.figure(figsize=(10, 8))
# sns.heatmap(spearman_corr, linewidth=1, annot=True, annot_kws={'size': 10})
# plt.title('Spearman Correlation', fontsize=25)
# plt.show()

# plt.figure(figsize=(10, 8))
# sns.heatmap(kendall_corr, linewidth=1, annot=True, annot_kws={'size': 10})
# plt.title('Kendall Correlation', fontsize=25)
# plt.show()

# visualization using yellow-brick library
features = clean_data_numeric.drop('Price', axis=1)
target = clean_data_numeric['Price']
feature_names = list(features.columns)

# visualizer = FeatureCorrelation(labels=feature_names, method='pearson')
# visualizer.fit(features, target)
# visualizer.poof()

# visualizer = FeatureCorrelation(method='mutual_info-regression', feature_names=feature_names, sort=True)
# visualizer.fit(features, target)
# visualizer.poof()
