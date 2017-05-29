import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from plot_test import generateSimulatedDimensionalityReductionData
from sklearn import preprocessing
import matplotlib.pyplot as plt
from ZIFA import ZIFA,block_ZIFA

n = 200
d = 20
k = 2
sigma = .3
n_clusters = 3
decay_coef = .1 # 0.79

X, Y, Z, ids = generateSimulatedDimensionalityReductionData(n_clusters, n, d, k, sigma, decay_coef)
gene_matrix = np.copy(Y)
np.random.shuffle(np.transpose(gene_matrix))
gene_matrix = gene_matrix.T
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
gene_matrix = min_max_scaler.fit_transform(gene_matrix)

encoder = nn.Sequential(nn.Linear(20,2), nn.Sigmoid())
decoder = nn.Sequential(nn.Linear(2,20), nn.Sigmoid())
autoencoder = nn.Sequential(encoder, decoder)

def next_batch(M, ind, mb_size, n):
    if ind + mb_size >= n:
        diff = ind + mb_size - n
        fh = M[:, 0:diff]
        sh = M[:, ind:]
        return np.dstack((fh,sh))
    else:
        return M[:, ind:ind+mb_size]

optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
for it in range(1, 10000):
    ind = it%n
    # X = sample_X(mb_size)
    X = gene_matrix[:,ind]
    X = Variable(torch.from_numpy(X).float()).view(1, d)
    X_sample = autoencoder(X)
    recon_loss = F.binary_cross_entropy(X_sample, X)
    if it%100 == 0:
        print recon_loss.data[0]
    optimizer.zero_grad()
    recon_loss.backward()
    optimizer.step()

output = []
for i in range(n):
    X = gene_matrix[:,i]
    print "original\n", i, X
    X = Variable(torch.from_numpy(X).float()).view(1,d)
    out = encoder(X).data.numpy()[0]
    output.append(out)
    print "reconstructed\n", i, autoencoder(X).data.numpy()[0]
    print"\n\n"
output = np.array([x.tolist() for x in output])

def plot(X=output, Y=Y, Z=Z, ids=ids):
    colors = ['red', 'blue', 'green']
    cluster_ids = sorted(list(set(ids)))
    for id in cluster_ids:
        plt.scatter(X[ids == id, 0], X[ids == id, 1], color = colors[id - 1], s = 4)
        plt.title('True Latent Positions\nFraction of Zeros %2.3f' % (Y == 0).mean())
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
    plt.show()


plot()
