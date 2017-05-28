import numpy as np
import random
from copy import deepcopy
import matplotlib as plt
from matplotlib.pyplot import *

random.seed(35)
np.random.seed(32)

def generateSimulatedDimensionalityReductionData(n_clusters, n, d, k, sigma, decay_coef):
    """
        generates data with multiple clusters.
        Checked.
        """
    mu = 3
    range_from_value = .1

    if n_clusters == 1:
            Z = np.random.multivariate_normal(mean = np.zeros([k,]), cov = np.eye(k), size = n).transpose()
            cluster_ids =  np.ones([n,])
    else:
            Z = np.zeros([k, n])
            cluster_ids = np.array([random.choice(range(n_clusters)) for i in range(n)])
    for id in list(set(cluster_ids)):
            idxs = cluster_ids == id
            cluster_mu = (np.random.random([k,]) - .5) * 5
            Z[:, idxs] = np.random.multivariate_normal(mean = cluster_mu, cov = .05 * np.eye(k), size = idxs.sum()).transpose()

    A = np.random.random([d, k]) - .5
    mu = np.array([(np.random.uniform() * range_from_value * 2 + (1 - range_from_value)) * mu for i in range(d)])
    sigmas = np.array([(np.random.uniform() * range_from_value * 2 + (1 - range_from_value)) * sigma for i in range(d)])
    noise = np.zeros([d, n])
    for j in range(d):
            noise[j, :] = mu[j] + np.random.normal(loc = 0, scale = sigmas[j], size = n)
    X = (np.dot(A, Z) + noise).transpose()
    Y = deepcopy(X)
    Y[Y < 0] = 0
    rand_matrix = np.random.random(Y.shape)

    cutoff = np.exp(-decay_coef * (Y ** 2))
    zero_mask = rand_matrix < cutoff
    Y[zero_mask] = 0
    print('Fraction of zeros: %2.3f; decay coef: %2.3f' % ((Y == 0).mean(), decay_coef))
    return X, Y, Z.transpose(), cluster_ids

n = 200
d = 20
k = 2
sigma = .3
n_clusters = 3
decay_coef = .023 # 0.79
# decay_coef = .1 # 0.64
# decay_coef = .34 # 0.07
X, Y, Z, ids = generateSimulatedDimensionalityReductionData(n_clusters, n, d, k, sigma, decay_coef)

def plot(X=X, Y=Y, Z=Z, ids=ids):
    colors = ['red', 'blue', 'green']
    cluster_ids = sorted(list(set(ids)))
    for id in cluster_ids:
        scatter(Z[ids == id, 0], Z[ids == id, 1], color = colors[id - 1], s = 4)
        title('True Latent Positions\nFraction of Zeros %2.3f' % (Y == 0).mean())
        xlim([-4, 4])
        ylim([-4, 4])
    show()

