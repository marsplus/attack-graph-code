import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pylab as plt
from threat_model import Threat_Model
from spectrum_attack import spectrum_attack

import torch

np.random.seed(1)


n = 128
G = nx.watts_strogatz_graph(n, 10, 0.2)
adj = nx.adjacency_matrix(G).todense()

alpha1, alpha2, alpha3 = 0.6, 0, 0.4

center = np.random.choice(range(G.order()))
S = list(G.neighbors(center)) + [center]
S_prime = list(set(G.nodes()) - set(S))
S = torch.LongTensor(S)
S_prime = torch.LongTensor(S_prime)

# haven't figured out how to take budget constraint into account
budget = 1.5
learning_rate = 1e-1
Alpha = [alpha1, alpha2, alpha3]
Attacker = Threat_Model(S, S_prime, Alpha, budget, learning_rate, G)
opt = torch.optim.SGD(Attacker.parameters(), lr=learning_rate)

ret = {'lambda_1': [], 'lambda_1_S': [], 'centrality': []}
Losses = []
while True:
    Loss = Attacker()
    Losses.append(Loss.item())
    lambda1_S, lambda1, normalizedCut = Attacker.getRet()
    ret['lambda_1'].append(lambda1.item())
    ret['lambda_1_S'].append(lambda1_S.item())
    ret['centrality'].append(normalizedCut.item())

    opt.zero_grad()
    Loss.backward()

    budget_this_step = Attacker.get_step_budget()
    current_used_budget = Attacker.get_used_budget()
    if current_used_budget + budget_this_step <= budget:
        opt.step()
        Attacker.update_used_budget(budget_this_step)
    else:
        break
                
ret = pd.DataFrame(ret)