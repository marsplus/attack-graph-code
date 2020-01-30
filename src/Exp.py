import torch
import pickle
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pylab as plt
from threat_model import Threat_Model
from spectrum_attack import spectrum_attack
from networkx.algorithms.community import greedy_modularity_communities
np.random.seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--numExp', type=int, default=1,
                    help='numExp')
parser.add_argument('--id', type=int, default=1,
                    help='parallel id')
parser.add_argument('--graph_type', type=str, default='BA',
                    help='graph type')

args = parser.parse_args() 

# check if there any hub in the graph
def exist_hubs(graph):
    maxDegreeNode = max(dict(graph.degree()), key=lambda x: dict(graph.degree())[x])
    maxDegree = graph.degree(maxDegreeNode)
    avgDegree = np.mean(list(dict(graph.degree()).values()))

    if maxDegree >= avgDegree**2:
        return (True, maxDegreeNode)
    else:
        return (False, None)


def select_comm(graph):
    comm = list(np.random.choice(list(greedy_modularity_communities(graph))))
    assert(len(comm) != 0)
    return comm


def gen_graph(graph_type):
    if graph_type == 'BA':
        G = nx.barabasi_albert_graph(n, 3)
    elif graph_type == 'Small-World':
        G = nx.watts_strogatz_graph(n, 10, 0.2)
    elif graph_type == 'Email':
        G = nx.read_edgelist('../data/email-Eu-core.txt', nodetype=int, data=(('time', float), ))
        mapping = {item: idx for idx, item in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
    elif graph_type == 'Stoc-Block':
        sizes = [25, 50, 75, 100, 125]
        num_c = len(sizes)
        within_p = 0.1
        out_p = 0.001
        probs = np.identity(num_c) * within_p + np.ones((num_c, num_c)) * out_p - np.identity(num_c) * out_p
        G = nx.stochastic_block_model(sizes, probs)

        # get the largest connected component
        comps = nx.connected_components(G)
        comp_max_idx = max(comps, key=lambda x: len(x))
        G = G.subgraph(comp_max_idx)
        mapping = {item: idx for idx, item in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
    return G


def exec_attack():
    lambda1_S_tmp = []
    centrality_tmp = []
    iter_cnt = 0
    while True:
        iter_cnt += 1
        Loss = Attacker()
        lambda1_S, lambda1, normalizedCut = Attacker.getRet()
        lambda1_S_tmp.append(lambda1_S)
        centrality_tmp.append(normalizedCut)

        opt.zero_grad()
        Loss.backward()

        budget_this_step = Attacker.get_step_budget()
        current_used_budget = Attacker.get_used_budget()

        if current_used_budget + budget_this_step <= Attacker_budget:
            opt.step()
            Attacker.update_used_budget(budget_this_step)
            current_used_budget = Attacker.get_used_budget()
        else:
            break

        assert(Attacker.check_constraint())
    
    lambda1_S_increase_ratio = (lambda1_S_tmp[-1] - lambda1_S_tmp[0]) / lambda1_S_tmp[0]
    lambda1_S_increase_ratio = lambda1_S_increase_ratio.detach().numpy().squeeze()

    centrality_increase_ratio = (centrality_tmp[-1] - centrality_tmp[0]) / centrality_tmp[0]
    centrality_increase_ratio = centrality_increase_ratio.detach().numpy().squeeze()

    utility = Attacker.get_utility()
    utility = utility.detach().numpy().squeeze()
    return (lambda1_S_increase_ratio, centrality_increase_ratio, utility)


# parameters
n = 375
alpha_1, alpha_2, alpha_3 = 0, 0, 1
Alpha = [alpha_1, alpha_2, alpha_3]
learning_rate = 0.1

graph_data = []
for i in range(args.numExp):
    G = gen_graph(args.graph_type)
    assert(nx.is_connected(G))
    adj = nx.adjacency_matrix(G).todense()

    # center = np.random.choice(range(G.order()))
    # S = list(G.neighbors(center)) + [center]
    S = select_comm(G)
    # S = list(G.neighbors(hub_id))
    S_prime = list(set(G.nodes()) - set(S))
    S = torch.LongTensor(S)
    S_prime = torch.LongTensor(S_prime)

    graph_data.append((G, adj, S, S_prime))


result = []
for budget_change_ratio in [0.01, 0.05, 0.1, 0.15, 0.2]:
# for budget_change_ratio in [0.1]:
    for exp in range(args.numExp):
        G, adj, S, S_prime = graph_data[exp]

        #alpha_1, alpha_2, alpha_3 = get_alpha(adj, S, S_prime)
        #print("a1: {:.4f}      a2: {:.4f}      a3: {:.4f}".format(alpha_1.item(), alpha_2.item(), alpha_3.item()))

        Attacker = Threat_Model(S, S_prime, Alpha, budget_change_ratio, learning_rate, G)
        Attacker_budget = Attacker.get_budget()
        opt = torch.optim.SGD(Attacker.parameters(), lr=learning_rate)
        lambda1_ret, centrality_ret, utility_ret = exec_attack()

        graph_size = len(G) 
        S_size = len(S)
        d_avg_S = np.mean([G.degree(i) for i in S.numpy()])

        result.append((lambda1_ret, centrality_ret, utility_ret, S_size, d_avg_S, graph_size, budget_change_ratio))
        print("Budget: {:.2f}%    Exp: {}    lambda1_increase_ratio: {:.4f}%    centrality_increase_ratio: {:.4f}%    \
               utility: {}\n"
            .format(budget_change_ratio*100, exp, lambda1_ret*100, centrality_ret*100, utility_ret))
    print('*' * 80)


# with open('../result/{}_30-10-60_numExp_{}.p'.format(args.graph_type, args.numExp), 'wb') as fid:
#     pickle.dump(result, fid)


