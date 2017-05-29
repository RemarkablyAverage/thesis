import torch
import torch.nn
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from plot_test import generateSimulatedDimensionalityReductionData
from sklearn import preprocessing
from ZIFA import ZIFA,block_ZIFA

n = 200
d = 20
k = 2
sigma = .3
n_clusters = 3
decay_coef = .5 # 0.79

X, Y, Z, ids = generateSimulatedDimensionalityReductionData(n_clusters, n, d, k, sigma, decay_coef)
gene_matrix = np.copy(Y)
# np.random.shuffle(np.transpose(gene_matrix))
gene_matrix = gene_matrix.T
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
gene_matrix = min_max_scaler.fit_transform(gene_matrix)

def create_norm_lookup(gene_matrix):
    ret = dict([])
    for i in range(gene_matrix.shape[1]):
        ret[i] = (np.std(gene_matrix[:,i]), np.mean(gene_matrix[:,i]))
    return ret

# norm = create_norm_lookup(gene_matrix)

# for k in norm.keys():
#     gene_matrix[:,k] = (gene_matrix[:,k] - norm[k][1])/norm[k][0]

mb_size = 1
z_dim = 200
X_dim = 20 #mnist.train.images.shape[1]
h_dim = 200
cnt = 0
lr = 1e-5


# Encoder
Q = torch.nn.Sequential(
    torch.nn.Linear(X_dim, h_dim, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, z_dim, bias=True)
)

# Decoder
P = torch.nn.Sequential(
    torch.nn.Linear(z_dim, h_dim, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, X_dim, bias=True),
    torch.nn.Sigmoid()
)

# Discriminator
D = torch.nn.Sequential(
    torch.nn.Linear(z_dim, h_dim, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(h_dim, 1, bias=True),
    torch.nn.Sigmoid()
)


def reset_grad():
    Q.zero_grad()
    P.zero_grad()
    D.zero_grad()

Q_solver = optim.Adam(Q.parameters(), lr=lr)
P_solver = optim.Adam(P.parameters(), lr=lr)
D_solver = optim.Adam(D.parameters(), lr=lr)

def next_batch(M, ind, mb_size, n):
    if ind + mb_size >= n:
        diff = ind + mb_size - n
        fh = M[:, 0:diff]
        sh = M[:, ind:]
        return np.dstack((fh,sh))
    else:
        return M[:, ind:ind+mb_size]

for it in range(30000):
    ind = it%d
    # X = sample_X(mb_size)
    # X = gene_matrix[:,ind:ind+mb_size]
    X = next_batch(gene_matrix, ind, mb_size, n)
    X = Variable(torch.from_numpy(X).float()).view(mb_size,d)

    """ Reconstruction phase """
    z_sample = Q(X)
    X_sample = P(z_sample)
    # print "actual", X.data.numpy()
    # print "recovered", X_sample.data.numpy()
    recon_loss = nn.binary_cross_entropy(X_sample, X)

    recon_loss.backward()
    P_solver.step()
    Q_solver.step()
    reset_grad()

    """ Regularization phase """
    # Discriminator
    z_real = Variable(torch.randn(mb_size, z_dim))
    z_fake = Q(X)

    D_real = D(z_real)
    D_fake = D(z_fake)

    D_loss = -torch.mean(torch.log(D_real) + torch.log(1 - D_fake))

    D_loss.backward()
    D_solver.step()
    reset_grad()

    # Generator
    z_fake = Q(X)
    D_fake = D(z_fake)

    G_loss = -torch.mean(torch.log(D_fake))

    G_loss.backward()
    Q_solver.step()
    reset_grad()

    # Print and plot every now and then
    if it % 1000 == 0:
        e = Q.parameters()
        print(e.next())
        print(e.next())
        print('Iter-{}; D_loss: {:.4}; G_loss: {:.4}; recon_loss: {:.4}'
              .format(it, D_loss.data[0], G_loss.data[0], recon_loss.data[0]))
        # print X_sample.data.numpy()

output = []
for i in range(n):
    X = gene_matrix[:,i]
    print "original\n", i,X
    X = Variable(torch.from_numpy(X).float()).view(1,d)
    # out = (Q(X).data.numpy()[0] - norm[i][1])/norm[i][0]
    out = Q(X).data.numpy()[0]
    output.append(out)
    print "reconstructed\n", i,P(Q(X)).data.numpy()[0]
    # print "reconstructed\n",(P(Q(X)).data.numpy()[0]- norm[i][1])/norm[i][0]
output = np.array([x.tolist() for x in output])

def plot(X=output, Y=Y, Z=Z, ids=ids):
    colors = ['red', 'blue', 'green']
    cluster_ids = sorted(list(set(ids)))
    for id in cluster_ids:
        plt.scatter(X[ids == id, 0], X[ids == id, 1], color = colors[id - 1], s = 4)
        plt.title('True Latent Positions\nFraction of Zeros %2.3f' % (Y == 0).mean())
        plt.xlim([-25, 25])
        plt.ylim([-25, 25])
    plt.show()


# plot()

# Zhat, params = block_ZIFA.fitModel(Y, 2)

# plot(X=Zhat)
