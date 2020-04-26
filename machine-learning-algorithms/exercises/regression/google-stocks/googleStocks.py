import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor, LinearRegression


# read and parse different movie-recommendations files
def read_file(filename, tbond=False):
    data = pd.read_csv(filename, sep=',', usecols=[0,5], names=['Date', 'Price'], header=0)
    if not tbond:
        returns = np.array(data["Price"][:-1], np.float) / np.array(data["Price"][1:], np.float) - 1
        data["Returns"] = np.append(returns, np.nan)
    if tbond:
        data["Returns"] = data["Price"] / 100
    data.index = data["Date"]
    data = data["Returns"][0:-1]
    return data


goog_data = read_file("./data/goog.csv")
nasdaq_data = read_file("./data/nasdaq.csv")
tbond_data = read_file("./data/tbond5yr.csv", tbond=True)

reg = SGDRegressor(eta0=0.1, max_iter=100000, fit_intercept=False)
reg.fit((nasdaq_data - tbond_data).values.reshape(-1, 1), (goog_data - tbond_data))
print("{:.4f}".format(reg.coef_[0]))
