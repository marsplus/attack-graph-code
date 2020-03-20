import torch
import numpy as np
import networkx as nx
import torch.nn as nn
import matplotlib.pyplot as plt

np.random.seed(1)


class Threat_Model(nn.Module):
    def __init__(self, S, S_prime, G):
        super(Threat_Model, self).__init__()
        self.S = S
        self.S_prime = S_prime
        self.numNodes = len(G)
        
        adj = nx.adjacency_matrix(G).todense()
        self.adj_tensor = torch.tensor(adj, dtype=torch.float32).requires_grad_(True)
        self.adj_tensor = nn.Parameter(self.adj_tensor)
        self.adj_tensor.register_hook(lambda x: x - torch.diag(torch.diag(x)))

    def forward(self):


        x_s = torch.zeros(self.numNodes)
        x_s[self.S] = 1

        eigVals, eigVecs = torch.symeig(self.adj_tensor, eigenvectors=True)
        self.v1 = eigVecs[:, -1]

        Loss = self.v1[self.S_prime].sum()
        return Loss


G = nx.Graph()
G.add_nodes_from([0, 1, 2, 3, 4])
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (1, 4)])
adj = nx.adjacency_matrix(G).todense()
S = torch.LongTensor([0, 1, 4])
S_prime = torch.LongTensor([2, 3])
Attacker = Threat_Model(S, S_prime, G)
opt = torch.optim.SGD(Attacker.parameters(), lr=0.1)

for i in range(20):
    Loss = Attacker()
    opt.zero_grad()
    Loss.backward()
    opt.step()
    print(Loss)


