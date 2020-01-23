import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pylab as plt
from threat_model import Threat_Model
from spectrum_attack import spectrum_attack

import torch

np.random.seed(1)

# generate synthetic network structures, for simulation purposes
n = 128
G = nx.watts_strogatz_graph(n, 10, 0.2)
adj = nx.adjacency_matrix(G).todense()

# trade-off parameters
alpha1, alpha2, alpha3 = 0.5, 0.2, 0.3

# randomly choose a subset of nodes, whose 
# induced subgraph becomes the subgraph we 
# are interested in
center = np.random.choice(range(G.order()))
S = list(G.neighbors(center)) + [center]
# S_prime = V \ S
S_prime = list(set(G.nodes()) - set(S))
# torch.LongTensor(): PyTorch way to say that S is some index set
S = torch.LongTensor(S)
S_prime = torch.LongTensor(S_prime)

# those input parameters as in Algorithm 1 in the write-up
budget = 2
learning_rate = 1e-1
Alpha = [alpha1, alpha2, alpha3]
Attacker = Threat_Model(S, S_prime, Alpha, budget, learning_rate, G)

# define an optimizer to solve our optimization problem
# Attacker.parameters() tells the optimizer what we wanna optimize
opt = torch.optim.SGD(Attacker.parameters(), lr=learning_rate)

# stores the experimental results
ret = {'lambda_1': [], 'lambda_1_S': [], 'centrality': []}
Losses = []

# implement Algorithm 1 in the write-up
while True:
    Loss = Attacker()
    Losses.append(Loss.item())
    lambda1_S, lambda1, normalizedCut = Attacker.getRet()

    ret['lambda_1'].append(lambda1.item())
    ret['lambda_1_S'].append(lambda1_S.item())
    ret['centrality'].append(normalizedCut.item())

    # opt.zero_grad(): clear accumulated gradients in the last step
    # Loss.backward(): compute the gradient of the Loss w.r.t the optimizer (the adjacency matrix in our case)
    opt.zero_grad()
    Loss.backward()

    # how much budget consumed in this step
    budget_this_step = Attacker.get_step_budget()

    # how much budget has been used until now
    current_used_budget = Attacker.get_used_budget()

    # whether the attacker's budget can afford it finish this step?
    if current_used_budget + budget_this_step <= budget:
        # opt.step(): actually adds the perturbation (gradients x learning rate) to the adjacency matrix
        opt.step()
        # attacker updates the amount of used budget
        Attacker.update_used_budget(budget_this_step)
    else:
        break


# for plotting purposes         
ret = pd.DataFrame(ret)
