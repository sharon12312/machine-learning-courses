import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

aapl = pd.read_csv('../data/AAPL.csv')
print(aapl.head())

# shuffle the data
aapl = aapl.sample(frac=1)
aapl.reset_index(inplace=True, drop=True)
print(aapl.head())

# data column type is 'object'
print(aapl.info())

# convert to datetime type
aapl['Date'] = pd.to_datetime(aapl['Date'])
print(aapl.head())
print(aapl.info())

# sort by date
aapl.sort_values(by='Date', inplace=True)
print(aapl.head())

# reset the index
aapl.reset_index(inplace=True, drop=True)
print(aapl.head())

# convert to python data time
aapl['Date'] = aapl['Date'].apply(lambda d: d.to_pydatetime())
print(aapl.info())

# extract the day of the week
# 1 - monday
# 2 - tuesday
# 3 - wednesday
# 4 - thursday
# 5 - friday
aapl['Weekday'] = aapl['Date'].apply(lambda d: d.isoweekday())
print(aapl.head(10))

# extract the year
extract_year = lambda x: datetime.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').year
aapl['Year'] = aapl['Date'].apply(extract_year)
print(aapl.sample(10))

# extract the month
extract_month = lambda x: datetime.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S').month
aapl['Month'] = aapl['Date'].apply(extract_month)
print(aapl.sample(10))

# visualization
# plt.figure(figsize=(10, 8))
# plt.plot(aapl['Date'], aapl['Close'])
# plt.xlabel('Date')
# plt.ylabel('AAPL stock price')
# plt.show()

# group by month
aapl_by_month = aapl.groupby(by='Month').mean()
print(aapl_by_month)

# plt.figure(figsize=(10, 8))
# plt.plot(aapl_by_month.index, aapl_by_month['Close'])
# plt.xlabel('Date')
# plt.ylabel('AAPL stock price')
# plt.show()
