import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pylab as plt
from threat_model import Threat_Model
from spectrum_attack import spectrum_attack
import torch

np.random.seed(1)

# generate synthetic network structures, for simulation purposes
# G = nx.read_edgelist('../data/email-Eu-core.txt', nodetype=int, data=(('time', float), ))
# mapping = {item: idx for idx, item in enumerate(G.nodes())}
# G = nx.relabel_nodes(G, mapping)

#G = nx.read_edgelist('../data/facebook_combined.txt', nodetype=int)
#G = nx.watts_strogatz_graph(n, 10, 0.2)
#G = nx.barabasi_albert_graph(size, 4)
#G = hyperbolic_graph(size, 5, 2.8)
#G = kronecker_graph(size, 0.05)

n = 128
G = nx.barabasi_albert_graph(n, 3)
adj = nx.adjacency_matrix(G).todense()

# trade-off parameters
alpha1, alpha2, alpha3 = 0.3, 0.05, 0.65

# randomly choose a subset of nodes, whose 
# induced subgraph becomes the subgraph we 
# are interested in
center = np.random.choice(range(G.order()))
S = list(G.neighbors(center)) + [center]
S_prime = list(set(G.nodes()) - set(S))
# torch.LongTensor(): PyTorch way to say that S is some index set
S = torch.LongTensor(S)
S_prime = torch.LongTensor(S_prime)

# input parameters for  Algorithm 1 as in the write-up
budget_change_ratio = 0.1
learning_rate = 0.1
Alpha = [alpha1, alpha2, alpha3]
Attacker = Threat_Model(S, S_prime, Alpha, budget_change_ratio, learning_rate, G)
Attacker_budget = Attacker.get_budget()

# define an optimizer to solve our optimization problem
# Attacker.parameters() tells the optimizer what we wanna optimize
# Note: we use SGD optimizer here. Using other optimizers (e.g., Adam) 
# may cause unexpected issues.
opt = torch.optim.SGD(Attacker.parameters(), lr=learning_rate)

# stores the experimental results
ret = {'lambda_1': [], 'lambda_1_S': [], 'centrality': []}
Losses = []

# implement Algorithm 1 in the write-up
stats = []
for i in range(1):
    while True:
        Loss = Attacker()
        Losses.append(Loss.item())
        lambda1_S, lambda1, normalizedCut = Attacker.getRet()

        ret['lambda_1'].append(lambda1.item())
        ret['lambda_1_S'].append(lambda1_S.item())
        ret['centrality'].append(normalizedCut.item())

        # opt.zero_grad(): clear accumulated gradients in the last step
        # Loss.backward(): compute the gradient of the Loss w.r.t the optimizer (the adjacency matrix in our case)
        # A correct order to run the optimizer: opt.zero_grad() -> Loss.backward() -> opt.step()
        opt.zero_grad()
        Loss.backward()

        # how much budget consumed in this step
        budget_this_step = Attacker.get_step_budget()

        # how much budget has been used until now
        current_used_budget = Attacker.get_used_budget()

        # whether the attacker's budget can afford it finish this step?
        if current_used_budget + budget_this_step <= Attacker_budget:
            # opt.step(): actually adds the perturbation (gradients x learning rate) to the adjacency matrix
            opt.step()
            # attacker updates the amount of used budget
            Attacker.update_used_budget(budget_this_step)
            current_used_budget = Attacker.get_used_budget()
        else:
            break

        lambda1_S_inc = (ret['lambda_1_S'][-1] - ret['lambda_1_S'][0]) / ret['lambda_1_S'][0]
        centrality_inc = (ret['centrality'][-1] - ret['centrality'][0]) / ret['centrality'][0] 

        S_size = len(S)
        d_avg_S = np.mean([G.degree(i) for i in S.numpy()])

    stats.append((lambda1_S_inc, centrality_inc, S_size, d_avg_S))
    print("Lambda1_S_inc: {:.4f}%      centrality_inc: {:.4f}%      S_size: {}      Avg. Degree: {:.4f}".format(\
        lambda1_S_inc*100, centrality_inc*100, S_size, d_avg_S))

