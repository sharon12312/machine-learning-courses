import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC

desired_width = 320
desired_columns = 10
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', desired_columns)


# check whether a given value is a missing value, if yes change it to NaN
def to_num(cell):
    try:
        return np.float(cell)
    except:
        return np.nan


# apply missing value check to a column/panda series
def series_to_num(series):
    return series.apply(to_num)


# convert the label string to integer 1/0
def to_label(label):
    if label == "ad.":
        return 1
    else:
        return 0


# return nice name for the label integer
def to_label_string(label):
    if label == 1:
        return "Ad"
    else:
        return "None Ad"


data = pd.read_csv("./movie-recommendations/ad.movie-recommendations", sep=',', header=None, low_memory=False)

train_data = data.iloc[0:,0:-1].apply(series_to_num)
train_data = train_data.dropna()

train_labels = data.iloc[train_data.index, -1].apply(to_label)

clf = LinearSVC()
clf.fit(train_data[100:2300], train_labels[100:2300])

result = clf.predict(train_data.iloc[12].values.reshape(1,-1))  # ad
print(to_label_string(result[0]))

result = clf.predict(train_data.iloc[-1].values.reshape(1,-1))  # none ad
print(to_label_string(result[0]))

