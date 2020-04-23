import implicit
import pandas as pd
import heapq
from scipy.sparse import coo_matrix

data = pd.read_csv("./data/u.data", sep='\t', header=None, usecols=[0,1,2], names=['userId', 'itemId', 'rating'])

# data structure:
# userId | itemId | rating
#    1   |   10   |   3
#    2   |   20   |   1
#    3   |   30   |   2

# fetch user column
data['userId'] = data['userId'].astype("category")

# fetch item column
data['itemId'] = data['itemId'].astype("category")

# create new metrics with axis x of users and axis y with items
rating_matrix = coo_matrix((data['rating'].astype("float"),
                            (data['itemId'].cat.codes.copy(),
                             data['userId'].cat.codes.copy())))

# ==> result of the matrix:
#
#        10      20      30   <--items axis
#  1 |   3    |   ?    |   ?
#  2 |   ?    |   1    |   ?
#  3 |   ?    |   ?    |   2
#
#  ^
#  |
# users axis

# create user and item factors using implicit
user_factors, item_factors = implicit.alternating_least_squares(rating_matrix, factors=10, regularization=0.01)

# user_factors will be:
#
#       factor1  | factor2   | factor3
#   1      x     |    y      |    z
#   2     ...    |   ...     |    ...
#  ...
#
# items_factors will be
#
#            10   |  20  |  30
#   factor1   x   | ...  |  ...
#   factor2   y   | ...  |  ...
#     ...

# dot return all the columns which contains all the factors,
# i.e. it get all the items for userId: 196
user196 = item_factors.dot(user_factors[196])

# get the 3 largest items of this specific user
result = heapq.nlargest(3, range(len(user196)), user196.take)

print(result)