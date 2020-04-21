import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn import datasets
from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding

X, color = datasets.make_swiss_roll(n_samples=2000)
print(X.shape)
print(color.shape)

# visualization
# X = pd.DataFrame(X)
# ax = plt.subplots(figsize=(8, 8))
# ax = plt.axes(projection='3d')
# ax.scatter3D(X[0], X[1], X[2], c=color, cmap=plt.cm.Spectral)
# plt.show()


def apply_manifold_learning(X, method):
    X = method.fit_transform(X)
    print('New shape of X: ', X.shape)

    X = pd.DataFrame(X)
    plt.subplots(figsize=(8, 8))
    plt.axis('equal')

    plt.scatter(X[0], X[1], c=color, cmap=plt.cm.Spectral)
    plt.xlabel('X[0]')
    plt.ylabel('X[1]')
    plt.show()

    return method


# multidimensional scaling - reduce dimensionality while trying to preserve the distance between instances
# n_components=2 means 2 dimensions
# mds = apply_manifold_learning(X, MDS(n_components=2, metric=False))  # not good
# mds = apply_manifold_learning(X, MDS(n_components=2, metric=True))  # very good

# locally linear embedding - measure how each instance relates to its closest neighbors
# lle = apply_manifold_learning(X, LocallyLinearEmbedding(n_neighbors=15, n_components=2, method='standard'))  # good
# lle = apply_manifold_learning(X, LocallyLinearEmbedding(n_neighbors=15, n_components=2, method='hessian'))  # very good
