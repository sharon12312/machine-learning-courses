import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from yellowbrick.target import FeatureCorrelation
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.feature_selection import RFE
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


house_data = pd.read_csv('../data/HousingData.csv')
house_data = house_data.drop(['B', 'LSTAT'], axis=1)
print(house_data.head())
print(house_data.shape)

col_names = ['CrimeRate', 'ZonedRatio', 'IndusRatio',
             'AlongRiver', 'NO2Level', 'RoomsPerHouse',
             'OldHomeRatio', 'DisFromCenter', 'RoadAccessIndex',
             'PropTaxRate', 'PupilTeacherRatio', 'MedianHomeValue']

house_data.columns = col_names
house_data = house_data.replace('NA', np.nan)

print(house_data.isnull().sum().sort_values(ascending=False))
data = house_data.fillna(house_data.mean())
print(data.isnull().sum())

features = data.drop('MedianHomeValue', axis=1)
target = data['MedianHomeValue']

# feature_names = list(features.columns)
# visualizer = FeatureCorrelation(labels=feature_names)
# visualizer.fit(features, target)
# visualizer.poof()

# ---------------------
# first approach

select_univariate = SelectKBest(f_regression, k=5).fit(features, target)
feature_mask = select_univariate.get_support()
print(features.columns[feature_mask])

print(pd.DataFrame({'FeatureName': features.columns,'Score': select_univariate.scores_})
      .sort_values(by='Score', ascending=False))
uni_df = pd.DataFrame({'Univariate Method': features.columns[feature_mask]})
print(uni_df)

# ---------------------
# second approach

linear_regression = LinearRegression()
rfe = RFE(estimator=linear_regression, n_features_to_select=5, step=1)
rfe.fit(features, target)
rfe_features = features.columns[rfe.support_]
print(rfe_features)
print(pd.DataFrame({'FeatureName': features.columns, 'Rank': rfe.ranking_}).sort_values(by='Rank'))
rfe_df = pd.DataFrame({'RFE Method': rfe_features})
print(rfe_df)

# ---------------------
# third approach (backward & forward)

# feature_selector = SequentialFeatureSelector(LinearRegression(), k_features=5, forward=False,
#                                              scoring='neg_mean_squared_error', cv=4)
# feature_filtered = feature_selector.fit(features, target)
# backward_features = list(feature_filtered.k_feature_names_) ???
# print(backward_features)
# print(pd.DataFrame({'Backward Method': backward_features}))

# feature_selector = SequentialFeatureSelector(LinearRegression(), k_features=5, forward=True,
#                                              scoring='neg_mean_squared_error', cv=4)
# feature_filtered = feature_selector.fit(features, target)
# forward_features = list(feature_filtered.k_feature_names_)
# print(forward_features)
# print(pd.DataFrame({'Forward Method': forward_features}))

# ----------------------
# forth approach

lasso = Lasso(alpha=1.0)
lasso.fit(features, target)
lasso_coef = pd.DataFrame({'Feature': features.columns,
                           'LassoCoef': lasso.coef_}).sort_values(by='LassoCoef', ascending=False)
print(lasso_coef)
lasso_coef['LassoCoef'] = abs(lasso_coef['LassoCoef'])
print(lasso_coef.sort_values(by='LassoCoef', ascending=False))

lasso_df = lasso_coef.sort_values(by='LassoCoef', ascending=False).head(5)
lasso_df = pd.DataFrame({'Lasso Method': lasso_df['Feature'].values})
print(lasso_df)

# ----------------------
# final selection

comp_selected_col_df = [uni_df, rfe_df, lasso_df]
final_df = pd.concat(comp_selected_col_df, axis=1)
print(final_df)

# ----------------------

result = []


def best_score(name, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    linear_model = LinearRegression(normalize=True).fit(X_train, y_train)

    print(name)
    print('Training score:', linear_model.score(X_train, y_train))

    y_pred = linear_model.predict(X_test)
    print('r2_score:', r2_score(y_test, y_pred))


best_score('Univariate', features[final_df['Univariate Method'].values], target)
best_score('Recursive', features[final_df['RFE Method'].values], target)
best_score('Lasso', features[final_df['Lasso Method'].values], target)

# => Recursive columns gets the best score for our model
