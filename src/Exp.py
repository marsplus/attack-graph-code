import torch
import pickle
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
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


# check if there any hub in the graph
def exist_hubs(graph):
    maxDegreeNode = max(dict(graph.degree()), key=lambda x: dict(graph.degree())[x])
    maxDegree = graph.degree(maxDegreeNode)
    avgDegree = np.mean(list(dict(graph.degree()).values()))

    if maxDegree >= avgDegree**2:
        return (True, maxDegreeNode)
    else:
        return (False, None)


# run a community detection algorithm, then
# randomly pick one community as S. 
# note that the size of S is usually less than 100
def select_comm(graph):
    all_comms = list(greedy_modularity_communities(graph))
    comm = list(np.random.choice(all_comms))
    #while len(comm) > 100:
    #    comm = list(np.random.choice(all_comms))
    assert(len(comm) != 0)
    return comm


# generate synthetic graphs
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


# execute the attack
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



# parameters for running experiments
n = 375
alpha_1, alpha_2, alpha_3 = 0.3, 0.1, 0.6
Alpha = [alpha_1, alpha_2, alpha_3]
learning_rate = 0.1


# generate the graphs
# each randomly generated graph is associated with 
# a randomly picked set S
graph_data = []
for i in range(args.numExp):
    G = gen_graph(args.graph_type)
    assert(nx.is_connected(G))
    adjacency_matrix = nx.adjacency_matrix(G).todense()

    # center = np.random.choice(range(G.order()))
    # S = list(G.neighbors(center)) + [center]
    S = select_comm(G)
    # S = list(G.neighbors(hub_id))
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

# run the attack, with varying budgets
result = []
Deg_dist_result = {0.01: None, 0.05: None, 0.1: None, 0.15: None, 0.2: None}
eig_adj_result = {0.01: None, 0.05: None, 0.1: None, 0.15: None, 0.2: None}
eig_laplacian_result = {0.01: None, 0.05: None, 0.1: None, 0.15: None, 0.2: None}

for budget_change_ratio in [0.01, 0.05, 0.1, 0.15, 0.2]:
#for budget_change_ratio in [0.01]:
    Deg_dist = pd.DataFrame()
    eig_adj = pd.DataFrame()
    eig_laplacian = pd.DataFrame()

    for exp in range(args.numExp):
        G, adj, S, S_prime = graph_data[exp]

        Attacker = Threat_Model(S, S_prime, Alpha, budget_change_ratio, learning_rate, G)
        Attacker_budget = Attacker.get_budget()
        opt = torch.optim.SGD(Attacker.parameters(), lr=learning_rate)
        lambda1_ret, centrality_ret, utility_ret = exec_attack()

        graph_size = len(G) 
        S_size = len(S)
        d_avg_S = np.mean([G.degree(i) for i in S.numpy()])

        # concatenate degree distribution
        deg_d, eig_A, eig_L = get_degree_dist_and_spectrum()
        deg_d.columns = ['original', 'attacked']
        if_detected = stats.ttest_ind(deg_d['original'], deg_d['attacked'])[1] <= 0.05
        # Deg_dist = pd.concat([Deg_dist, deg_d])
        # eig_adj  = pd.concat([eig_adj, eig_A])
        # eig_laplacian = pd.concat([eig_laplacian, eig_L])

        result.append((lambda1_ret, centrality_ret, utility_ret, S_size, d_avg_S, graph_size, if_detected, budget_change_ratio))
        print("Budget: {:.2f}%    Exp: {}    lambda1_increase_ratio: {:.4f}%    centrality_increase_ratio: {:.4f}%    utility: {}    if_detected: {}\n".format(\
                budget_change_ratio*100, exp, lambda1_ret*100, centrality_ret*100, utility_ret, if_detected))
    # Deg_dist.index = range(len(Deg_dist))
    # Deg_dist.columns = ['original', 'attacked']
    # Deg_dist_result[budget_change_ratio] = Deg_dist

    # eig_adj.index = range(len(eig_adj))
    # eig_adj.columns = ['original', 'attacked']
    # eig_adj_result[budget_change_ratio] = eig_adj

    # eig_laplacian.index = range(len(eig_laplacian))
    # eig_laplacian.columns = ['original', 'attacked']
    # eig_laplacian_result[budget_change_ratio] = eig_laplacian
    print('*' * 80)


with open('../result/{}_30-10-60_numExp_{}.p'.format(args.graph_type, args.numExp), 'wb') as fid:
    pickle.dump(result, fid)

# with open('../result/{}_30-10-60_numExp_{}_Deg_distribution.p'.format(args.graph_type, args.numExp), 'wb') as fid:
#     pickle.dump(Deg_dist_result, fid)

# with open('../result/{}_30-10-60_numExp_{}_adj_spectrum.p'.format(args.graph_type, args.numExp), 'wb') as fid:
#     pickle.dump(eig_adj_result, fid)

# with open('../result/{}_30-10-60_numExp_{}_laplacian_spectrum.p'.format(args.graph_type, args.numExp), 'wb') as fid:
#     pickle.dump(eig_laplacian_result, fid)


