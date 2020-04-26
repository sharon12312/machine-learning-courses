import pandas as pd
import datetime

data = pd.read_csv('data/123.csv')
print(data.head())

# remove unnecessary columns
del data['none']
del data['id']

print(data.head())

# convert to python data time
# data['date'] = data['date'].apply(lambda d: d.to_pydatetime())
data['date'] = pd.to_datetime(data['date'])

print(data.info())

# extract the year
extract_year = lambda x: datetime.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').year
data['year'] = data['date'].apply(extract_year)
# print(data.sample(10))

# extract the month
extract_month = lambda x: datetime.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').month
data['month'] = data['date'].apply(extract_month)
# print(data.sample(10))

# extract the day
extract_month = lambda x: datetime.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').day
data['day'] = data['date'].apply(extract_month)
# print(data.sample(10))

data = data[['date', 'day', 'month', 'year', '1', '2', '3']]
print(data.head())

del data['date']

data['numbers'] = data.apply(lambda row: row['1']*100 + row['2']*10 + row['3'], axis=1)

data.to_csv('./data/123-processed-new.csv', index=False)
# Done
