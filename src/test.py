import torch
import numpy as np
import networkx as nx
import torch.nn as nn
from utils import *
import numpy.linalg as LIN
import matplotlib.pyplot as plt

np.random.seed(1)


class Threat_Model(nn.Module):
    def __init__(self, S, S_prime, G, flag=True):
        super(Threat_Model, self).__init__()
        self.S = S
        self.S_prime = S_prime
        self.numNodes = len(G)
        self.flag = flag

        adj = nx.adjacency_matrix(G).todense()
        self.adj_tensor = torch.tensor(adj, dtype=torch.float32).requires_grad_(True)
        self.adj_tensor = nn.Parameter(self.adj_tensor)
        self.adj_tensor.register_hook(lambda x: x - torch.diag(torch.diag(x)))

    def forward(self):


        x_s = torch.zeros(self.numNodes)
        x_s[self.S] = 1

        #eigVals, eigVecs = torch.symeig(self.adj_tensor, eigenvectors=True)
        #self.v1 = eigVecs[:, -1]
        if self.flag:
            self.v1 = power_method(self.adj_tensor.data, 50)
            Loss = self.v1[self.S].sum()
        else:
            eigVals, eigVecs = torch.symeig(self.adj_tensor, eigenvectors=True)
            Loss = torch.max(eigVals) 


        return Loss

n = 10
G = nx.watts_strogatz_graph(n, 10, 0.2)
adj = nx.adjacency_matrix(G).todense()
S = np.random.choice(G.nodes(), 10, replace=False)
S_prime = list(set(G.nodes()) - set(S))
S = torch.LongTensor(S)
S_prime = torch.LongTensor(S_prime)
Attacker = Threat_Model(S, S_prime, G, True)
opt = torch.optim.SGD(Attacker.parameters(), lr=0.01)

for i in range(50):
    Loss = Attacker()
    opt.zero_grad()
    Loss.backward()
    opt.step()
    print(Loss)


