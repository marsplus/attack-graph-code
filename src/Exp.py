import time
import math
import torch
import pickle
import argparse
import numpy as np
from utils import *
import pandas as pd
import networkx as nx
from scipy import stats
import numpy.linalg as LIN
from model import Threat_Model
from collections import defaultdict
from networkx.algorithms.community import greedy_modularity_communities

np.random.seed(1)
parser = argparse.ArgumentParser()
parser.add_argument('--numExp', type=int, default=1,
                    help='numExp')
parser.add_argument('--id', type=int, default=1,
                    help='parallel id')
parser.add_argument('--graph_type', type=str, default='BA',
                    help='graph type')
parser.add_argument('--Alpha_id', type=int, default=1,
                    help='which set of parameters to use')

args = parser.parse_args()


# run a community detection algorithm, then
# randomly pick one community as S.
# note that the size of S is usually less than 100
def select_comm(graph, isEmail=False, mapping=None):
    if isEmail:
        # read into community info
        with open('../data/email-Eu-core-department-labels-cc.txt', 'r') as fid:
            f_label = fid.readlines()
        comm_to_nodes = {}
        for item in f_label:
            nodeID, commID = [int(i) for i in item.rstrip().split()]
            if commID not in comm_to_nodes:
                comm_to_nodes[commID] = [mapping[nodeID]]
            else:
                comm_to_nodes[commID].append(mapping[nodeID])
        comm_size = sorted([(key, len(comm_to_nodes[key])) for key in comm_to_nodes.keys()], key=lambda x: x[1])
        comm = comm_to_nodes[comm_size[math.floor(len(comm_size) * 0.5)][0]]
    else:
        all_comms = list(greedy_modularity_communities(graph))
        all_comms = sorted(all_comms, key=lambda x: len(x))
        comm = list(all_comms[math.floor(len(all_comms) * 0.6)])
        assert(len(comm) != 0)
    return comm


# generate synthetic graphs
def gen_graph(graph_type, graph_id=1):
    if graph_type == 'BA':
        G = nx.barabasi_albert_graph(n, 5)
    elif graph_type == 'Small-World':
        G = nx.watts_strogatz_graph(n, 10, 0.2)
    elif graph_type == 'Email':
        G = nx.read_edgelist('../data/email-Eu-core-cc.txt', nodetype=int)
    elif graph_type == 'Facebook':
        G = nx.read_edgelist('../data/facebook_combined.txt', nodetype=int)
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
    elif graph_type == 'BTER':
        G = nx.read_edgelist('../data/BTER_{:02d}.txt'.format(graph_id), nodetype=int)
    return G



# execute projection attack
def projection_attack():
    # optimize until some weights are about to
    # become negative
    while torch.all(Attacker.adj_tensor >= 0):
        Loss = Attacker()
        opt_Adam.zero_grad()
        Loss.backward()
        opt_Adam.step()

    # Projection step
    if not Attacker.check_constraint():
        print("start projection\n")
        # projection step
        Delta = Attacker.adj_tensor.data - Attacker.original_adj
        U, Sigma, V = torch.svd(Delta)
        Sigma_proj = L_inf_proj(Sigma, Attacker.get_budget())
        Delta_proj = U @ torch.diag(Sigma_proj) @ V.T
        # add projected Delta to adjacency matrix
        Attacker.adj_tensor.data = Attacker.original_adj + Delta_proj

    Attacker() # this updates all the statistics since adj_tensor has been updated
    lambda1_S, _, centrality = Attacker.getRet()
    lambda1_S_0, centrality_0 = Attacker.lambda1_S_original, Attacker.centrality_original
    lambda1_S_increase_ratio = (lambda1_S - lambda1_S_0) / lambda1_S_0
    centrality_increase_ratio = (centrality - centrality_0) / centrality_0
    utility = Attacker.get_utility()

    return (lambda1_S_increase_ratio.detach().numpy().squeeze(),
            centrality_increase_ratio.detach().numpy().squeeze(),
            utility.detach().numpy().squeeze())


# parameters for running experiments
n = 375
# Alpha = [0.4, 0, 0.6]
learning_rate = 0.1


result = defaultdict(list)
for budget_change_ratio in [0.01, 0.05, 0.1, 0.15, 0.2]:
    for i in range(args.numExp):
        G = gen_graph(args.graph_type, i)
        mapping = {item: idx for idx, item in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
        adj = nx.adjacency_matrix(G).todense()
        
        if args.graph_type == "Email":
            S = select_comm(G, True, mapping)
        else:
            S = select_comm(G)

        S_prime = list(set(G.nodes()) - set(S))
        S = torch.LongTensor(S)
        S_prime = torch.LongTensor(S_prime)

        ret = []
        # budget_change_ratio = 0.1
        for alpha_1 in np.arange(0.01, 0.99, 0.01):
            alpha_2 = 0
            alpha_3 = 1 - alpha_1 - alpha_2
            Alpha = [alpha_1, alpha_2, alpha_3]
            print("alpha_1: {:.4f}      alpha_2: {:.4f}     alpha_3: {:.4f}\n".format(alpha_1, alpha_2, alpha_3))

            Attacker = Threat_Model(S, S_prime, Alpha, budget_change_ratio, learning_rate, G)
            opt_Adam = torch.optim.Adam(Attacker.parameters(), lr=learning_rate)
            t1 = time.time()
            lambda1_ret, centrality_ret, utility_ret = projection_attack()
            print("Time: {:.4f}".format(time.time() - t1))

            graph_size = len(G) 
            S_size = len(S)
            d_avg_S = np.mean([G.degree(i) for i in S.numpy()])
            ret.append((utility_ret, lambda1_ret, centrality_ret, budget_change_ratio, Alpha))
            print("Budget: {:.2f}%     lambda1_increase_ratio: {:.4f}%    centrality_increase_ratio: {:.4f}%    utility: {}\n".format(\
                    budget_change_ratio*100, lambda1_ret*100, centrality_ret*100, utility_ret))
            print('*' * 80)
        ret_pos = [item for item in ret if (item[1] + item[2]) >= 0]
        if ret_pos:
            result[budget_change_ratio].append(max(ret_pos, key=lambda x: x[1] * x[2]))
        else:
            result[budget_change_ratio].append(max(ret, key=lambda x: x[1] * x[2]))


# with open('../result/{}_numExp_{}.p'.format(args.graph_type, args.numExp), 'wb') as fid:
#     pickle.dump(result, fid)

