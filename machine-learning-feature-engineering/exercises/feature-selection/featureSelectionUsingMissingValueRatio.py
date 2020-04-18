import pandas as pd

melb_data = pd.read_csv('./data/melb_data.csv')

print(melb_data.head())
print(melb_data.shape)

# presents the columns which contain null values
empty_col_cells = melb_data.isnull().sum().sort_values(ascending=False)
print(empty_col_cells.head(14))
# presents columns' percent
print((melb_data.isnull().sum() / melb_data.shape[0]).sort_values(ascending=False))

print(melb_data[['BuildingArea', 'YearBuilt', 'CouncilArea', 'Car']].head())
print(melb_data[['BuildingArea', 'YearBuilt', 'CouncilArea', 'Car']].isnull().sum())
print(melb_data[['BuildingArea', 'YearBuilt', 'CouncilArea', 'Car']].describe())

# fill null values of 'Car'
melb_data['Car'] = melb_data['Car'].fillna(melb_data['Car'].median())
print(melb_data[['BuildingArea', 'YearBuilt', 'CouncilArea', 'Car']].describe())
print(melb_data[['BuildingArea', 'YearBuilt', 'CouncilArea', 'Car']].isnull().sum())

# fill null values of 'CouncilArea'
print(melb_data[['BuildingArea', 'YearBuilt', 'CouncilArea', 'Car']].tail())
print(melb_data['CouncilArea'].mode())  # most common value
melb_data['CouncilArea'] = melb_data['CouncilArea'].fillna(melb_data['CouncilArea'].mode()[0])
print(melb_data[['BuildingArea', 'YearBuilt', 'CouncilArea', 'Car']].tail())
print(melb_data[['BuildingArea', 'YearBuilt', 'CouncilArea', 'Car']].isnull().sum())

# if the table has more than 30% missing values, then drop them from the table
print(melb_data.columns)
melb_data.dropna(thresh=int(melb_data.shape[0] * .7), axis=1, inplace=True)
print(melb_data.columns)

# check for more fields with null values
empty_col_cells = melb_data.isnull().sum().sort_values(ascending=False)
print(empty_col_cells.head(4))
print((melb_data.isnull().sum() / melb_data.shape[0]).sort_values(ascending=False))

# fill null values of the rest columns who needed
# for int values
melb_data['Propertycount'] = melb_data['Propertycount'].fillna(melb_data['Propertycount'].median())
melb_data['Postcode'] = melb_data['Postcode'].fillna(melb_data['Postcode'].median())
melb_data['Distance'] = melb_data['Distance'].fillna(melb_data['Distance'].median())
melb_data['Bathroom'] = melb_data['Bathroom'].fillna(melb_data['Bathroom'].median())
melb_data['Bedroom2'] = melb_data['Bedroom2'].fillna(melb_data['Bedroom2'].median())
melb_data['Longtitude'] = melb_data['Longtitude'].fillna(melb_data['Longtitude'].median())
melb_data['Lattitude'] = melb_data['Lattitude'].fillna(melb_data['Lattitude'].median())
melb_data['Price'] = melb_data['Price'].fillna(melb_data['Price'].median())

# for string values
melb_data['Regionname'] = melb_data['Regionname'].fillna(melb_data['Regionname'].mode()[0])

# check for more fields with null values
empty_col_cells = melb_data.isnull().sum()
print(empty_col_cells.head(4))
print((melb_data.isnull().sum() / melb_data.shape[0]).sort_values(ascending=False))

# write the new csv file
melb_data.to_csv('./data/melb_data_processed.csv', index=False)
