import pandas as pd
import numpy as np

df = pd.DataFrame(data= {
    'feature_1': [np.nan, 3, 6, 9, 12, 15, np.nan],
    'feature_2': [100, np.nan, 200, 300, np.nan, np.nan, 600],
    'feature_3': [1000, 500, 2000, 3000, 4000, 6000, 8000]
})

print(df)
print(df.isnull())
print(df.isnull().sum())

# get the value from the previous cell
# limit the numbers of missing values in the specific column
print(df.fillna(method='pad', limit=1))
print(df.fillna(method='pad', limit=2))

# fill by the next cell
print(df.fillna(method='bfill'))

# fill by the previous cell
print(df.fillna(method='ffill'))

# drop rows
print(df.dropna(axis=0))

# drop columns
print(df.dropna(axis=1))

# the threshold specifies the present number of rows which need to be non null values
print(df.dropna(thresh=int(df.shape[0] * .9), axis=1))
print(df.shape[0] * .9)

# fill with average's column
print(df['feature_1'].fillna(df['feature_1'].mean()))

# fill with the oriented values, e.g. 1, na, 3 -> will become 1,2,3
print(df['feature_2'].interpolate())

