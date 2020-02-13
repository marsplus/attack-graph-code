import time
import torch
import pickle
import argparse
import numpy as np
from utils import *
import pandas as pd
import networkx as nx
from scipy import stats
import numpy.linalg as LIN
import matplotlib.pylab as plt
from model import Threat_Model
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
parser.add_argument('--Alpha_id', type=int, default=1,
                    help='which set of parameters to use')

args = parser.parse_args()


# record original and attacked degree distributionis
def get_degree_dist_and_spectrum():
    A = torch.tensor(adj, dtype=torch.float32)
    D = torch.diag(torch.mm(A, torch.ones(len(A)).view(-1, 1)).squeeze())
    L = D - A

    attacked_L = Attacker.get_Laplacian()
    attacked_A = Attacker.get_attacked_adj()
    attacked_D = attacked_L + attacked_A

    degs = torch.diag(D).numpy()
    attacked_degs = torch.diag(attacked_D).numpy()
    data_degs = pd.DataFrame([degs, attacked_degs]).transpose()

    eigVals_A, _ = torch.symeig(A, eigenvectors=True)
    attacked_eigVals_A, _ = torch.symeig(attacked_A, eigenvectors=True)
    eigVals_A = eigVals_A.numpy()
    attacked_eigVals_A = attacked_eigVals_A.numpy()
    data_eig_A = pd.DataFrame([eigVals_A, attacked_eigVals_A]).transpose()

    eigVals_L, _ = torch.symeig(L, eigenvectors=True)
    attacked_eigVals_L, _ = torch.symeig(attacked_L, eigenvectors=True)
    eigVals_L = eigVals_L.numpy()
    attacked_eigVals_L = attacked_eigVals_L.numpy()
    data_eig_L = pd.DataFrame([eigVals_L, attacked_eigVals_L]).transpose()

    return (data_degs, data_eig_A, data_eig_L)


# run a community detection algorithm, then
# randomly pick one community as S.
# note that the size of S is usually less than 100
def select_comm(graph):
    all_comms = list(greedy_modularity_communities(graph))
    all_comms = [item for item in all_comms if len(item) < 100]

    if len(all_comms) == 0:
        all_comms = list(greedy_modularity_communities(graph))

    comm = list(np.random.choice(all_comms))
    assert(len(comm) != 0)
    return comm


# generate synthetic graphs
def gen_graph(graph_type):
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
    return G


# execute SGD attack
# projection when necessary
def exec_attack_SGD():
    Attacker_budget = Attacker.get_budget()
    Attacker()
    lambda1_S_0, lambda1_0, centrality_0 = Attacker.getRet()
    while True:
        Loss = Attacker()
        opt_SGD.zero_grad()
        Loss.backward()

        budget_this_step = Attacker.get_step_budget()
        current_used_budget = Attacker.get_used_budget()

        if current_used_budget + budget_this_step <= Attacker_budget:
            opt_SGD.step()
            Attacker.update_used_budget(budget_this_step)
            current_used_budget = Attacker.get_used_budget()
        else:
            break

    # # O(n^3)
    # if not Attacker.check_constraint():
    #     # projection step
    #     Delta = Attacker.adj_tensor.data - Attacker.original_adj
    #     U, Sigma, V = torch.svd(Delta)
    #     Sigma_proj = L_inf_proj(Sigma, Attacker.get_budget())
    #     Delta_proj = U @ torch.diag(Sigma_proj) @ V.T
    #     # compute results after projection
    #     Attacker.adj_tensor.data = Attacker.original_adj + Delta_proj

    Attacker()
    lambda1_S, lambda1, centrality = Attacker.getRet()
    lambda1_S_increase_ratio = (lambda1_S - lambda1_S_0) / lambda1_S_0
    centrality_increase_ratio = (centrality - centrality_0) / centrality_0
    utility = Attacker.get_utility()

    return (lambda1_S_increase_ratio.detach().numpy().squeeze(),
            centrality_increase_ratio.detach().numpy().squeeze(),
            utility.detach().numpy().squeeze())

    return (lambda1_S_increase_ratio, centrality_increase_ratio, utility)



# execute projection attack
def exec_attack_Projection():
    Attacker()
    lambda1_S_0, lambda1_0, centrality_0 = Attacker.getRet()

    # optimize until some weights are about to
    # become negative
    while torch.all(Attacker.adj_tensor >= 0):
    # for i in range(100):
        Loss = Attacker()
        opt_Adam.zero_grad()
        Loss.backward()
        opt_Adam.step()

    if not Attacker.check_constraint():
        print("start projection\n")
        # projection step
        Delta = Attacker.adj_tensor.data - Attacker.original_adj
        U, Sigma, V = torch.svd(Delta)

        Sigma_proj = L_inf_proj(Sigma, Attacker.get_budget())
        Delta_proj = U @ torch.diag(Sigma_proj) @ V.T

        # compute results after projection
        Attacker.adj_tensor.data = Attacker.original_adj + Delta_proj

    Attacker()
    lambda1_S, lambda1, centrality = Attacker.getRet()
    lambda1_S_increase_ratio = (lambda1_S - lambda1_S_0) / lambda1_S_0
    centrality_increase_ratio = (centrality - centrality_0) / centrality_0
    utility = Attacker.get_utility()

    return (lambda1_S_increase_ratio.detach().numpy().squeeze(),
            centrality_increase_ratio.detach().numpy().squeeze(),
            utility.detach().numpy().squeeze())


num_to_Alpha = {
    1: [0.3, 0.1, 0.6],
    2: [0.4, 0.1, 0.5],
    3: [0.45, 0.1, 0.45],
    4: [0.01, 0, 0.99],
}

# parameters for running experiments
n = 375
#Alpha = num_to_Alpha[args.Alpha_id]
Alpha = [0.5, 0.1, 1.5]
learning_rate = 0.01


# generate the graphs
# each randomly generated graph is associated with
# a randomly picked set S
graph_data = []
if args.graph_type == 'Facebook':
    G = gen_graph(args.graph_type)
    assert(nx.is_connected(G))
    adjacency_matrix = nx.adjacency_matrix(G).todense()

    all_comms = list(greedy_modularity_communities(G))
    # all_comms = [item for item in all_comms if len(item) > 200 and len(item) < 300]
    # all_comms = [item for item in all_comms if len(item) == 37]

    if len(all_comms) == 0:
        all_comms = list(greedy_modularity_communities(graph))

    for i in range(args.numExp):
        S = list(np.random.choice(all_comms))
        assert(len(S) != 0)
        S_prime = list(set(G.nodes()) - set(S))
        S = torch.LongTensor(S)
        S_prime = torch.LongTensor(S_prime)
        graph_data.append((G, adjacency_matrix, S, S_prime))
elif args.graph_type == 'Email':
    G = gen_graph(args.graph_type)
    mapping = {item: idx for idx, item in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    assert(nx.is_connected(G))
    adjacency_matrix = nx.adjacency_matrix(G).todense()

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

    # candidate communities with at least 3 people
    #cand_comm_ID = [key for key in comm_to_nodes.keys() if len(comm_to_nodes[key]) >= 50 and len(comm_to_nodes[key]) <= 100]
    cand_comm_ID = [key for key in comm_to_nodes.keys() if len(comm_to_nodes[key]) >= 20 and len(comm_to_nodes[key]) <= 50]
    for i in range(args.numExp):
        S = comm_to_nodes[np.random.choice(cand_comm_ID)]
        S_prime = list(set(G.nodes()) - set(S))
        S = torch.LongTensor(S)
        S_prime = torch.LongTensor(S_prime)
        graph_data.append((G, adjacency_matrix, S, S_prime))
else:
    for i in range(args.numExp):
        G = gen_graph(args.graph_type)
        assert(nx.is_connected(G))
        adjacency_matrix = nx.adjacency_matrix(G).todense()

        S = select_comm(G)
        S_prime = list(set(G.nodes()) - set(S))
        S = torch.LongTensor(S)
        S_prime = torch.LongTensor(S_prime)
        graph_data.append((G, adjacency_matrix, S, S_prime))



budget_to_string = {
    0.01: '1%',
    0.05: '5%',
    0.1:  '10%',
    0.15: '15%',
    0.2:  '20%'
}

Alpha_to_string = {
    1: '30-10-60',
    2: '40-10-50',
    3: '45-10-45',
    4: '1-0-99',
}


# run the attack, with varying budgets
result = []
Deg_dist_result = {0.01: None, 0.05: None, 0.1: None, 0.15: None, 0.2: None}
eig_adj_result = {0.01: None, 0.05: None, 0.1: None, 0.15: None, 0.2: None}
eig_laplacian_result = {0.01: None, 0.05: None, 0.1: None, 0.15: None, 0.2: None}

for budget_change_ratio in [0.01, 0.05, 0.1, 0.15, 0.2]:
# for budget_change_ratio in [0.1]:
    Deg_dist = pd.DataFrame()
    eig_adj = pd.DataFrame()
    eig_laplacian = pd.DataFrame()

    for exp in range(args.numExp):
        G, adj, S, S_prime = graph_data[exp]

        Attacker = Threat_Model(S, S_prime, Alpha, budget_change_ratio, learning_rate, G)
        opt_Adam = torch.optim.Adam(Attacker.parameters(), lr=learning_rate)
        #opt_SGD = torch.optim.SGD(Attacker.parameters(), lr=learning_rate)

        t1 = time.time()
        try:
            lambda1_ret, centrality_ret, utility_ret = exec_attack_Projection()
        except:
            print("Cannot execute attack\n")
            continue
        print("Time: {:.4f}".format(time.time() - t1))

        graph_size = len(G)
        S_size = len(S)
        d_avg_S = np.mean([G.degree(i) for i in S.numpy()])

        # concatenate degree distribution
        deg_d, eig_A, eig_L = get_degree_dist_and_spectrum()
        deg_d.columns = ['original', 'attacked']
        if_detected = stats.ttest_ind(deg_d['original'], deg_d['attacked'])[1] <= 0.05
        Deg_dist = pd.concat([Deg_dist, deg_d])
        eig_adj  = pd.concat([eig_adj, eig_A])
        eig_laplacian = pd.concat([eig_laplacian, eig_L])

        result.append((lambda1_ret, centrality_ret, utility_ret, S_size, d_avg_S, graph_size, if_detected, budget_change_ratio))
        print("Budget: {:.2f}%    Exp: {}    lambda1_increase_ratio: {:.4f}%    centrality_increase_ratio: {:.4f}%    utility: {}    if_detected: {}\n".format(\
                budget_change_ratio*100, exp, lambda1_ret*100, centrality_ret*100, utility_ret, if_detected))

    Deg_dist.index = range(len(Deg_dist))
    Deg_dist.columns = ['original', 'attacked']
    Deg_dist_result[budget_change_ratio] = Deg_dist

    eig_adj.index = range(len(eig_adj))
    eig_adj.columns = ['original', 'attacked']
    eig_adj_result[budget_change_ratio] = eig_adj

    eig_laplacian.index = range(len(eig_laplacian))
    eig_laplacian.columns = ['original', 'attacked']
    eig_laplacian_result[budget_change_ratio] = eig_laplacian

    print('*' * 80)


with open('../result/{}_{}_numExp_{}.p'.format(args.graph_type, "middle-quantile", args.numExp), 'wb') as fid:
   pickle.dump(result, fid)

with open('../result/{}_{}_numExp_{}_Deg_distribution.p'.format(args.graph_type, "middle-quantile", args.numExp), 'wb') as fid:
   pickle.dump(Deg_dist_result, fid)

with open('../result/{}_{}_numExp_{}_adj_spectrum.p'.format(args.graph_type, "middle-quantile", args.numExp), 'wb') as fid:
    pickle.dump(eig_adj_result, fid)

with open('../result/{}_{}_numExp_{}_laplacian_spectrum.p'.format(args.graph_type, "middle-quantile", args.numExp), 'wb') as fid:
    pickle.dump(eig_laplacian_result, fid)
