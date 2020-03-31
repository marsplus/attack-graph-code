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
from scipy import sparse
import numpy.linalg as LIN
from model import Threat_Model
import scipy.sparse.linalg as sLIN
from collections import defaultdict
from networkx.algorithms.community import greedy_modularity_communities

np.random.seed(1)
parser = argparse.ArgumentParser()
parser.add_argument('--numExp', type=int, default=1,
                    help='numExp')
parser.add_argument('--graph_type', type=str, default='BA',
                    help='graph type')
parser.add_argument('--mode', type=str, default='equalAlpha',
                    help='trade-off parameters mode')
parser.add_argument('--weighted_graph', type=bool, default=True,
                    help='weighted graphs or not')
args = parser.parse_args()


# parameters for running experiments
n = 375
learning_rate = 1
MAPPING = {
        'equalAlpha': [1/3, 1/3, 1/3],
        'alpha1=1': [1/3, 0, 0],
        'alpha2=0': [1/3, 0, 1/3],
        'alpha3=0': [1/3, 1/3, 0],
        'alpha3=1': [0, 0, 1/3],
        }



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
        comm = list(all_comms[math.floor(len(all_comms) * 0.5)])
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
        G.remove_edges_from(nx.selfloop_edges(G))
    elif graph_type == 'Facebook':
        G = nx.read_edgelist('../data/facebook_combined.txt', nodetype=int)
    elif graph_type == 'Stoc-Block':
        sizes = [25, 50, 100, 200]
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




def SGD_attack(Attacker, Optimizer, Iter=100):
    Attacker_budget = Attacker.get_budget()
    cnt = 0
    while True:
        Loss = Attacker()
        Optimizer.zero_grad()
        Loss.backward()

        budget_this_step = Attacker.get_step_budget()
        current_used_budget = Attacker.get_used_budget()

        if current_used_budget + budget_this_step <= Attacker_budget:
            Optimizer.step()
            Attacker.update_used_budget(budget_this_step)
            current_used_budget = Attacker.get_used_budget()
            cnt += 1
        else:
            break
    print("SGD iterations: {}".format(cnt))
    
    Attacker.adj_tensor.data[Attacker.adj_tensor.data < 0] = 0
    addedEdges = 0

    Attacker()
    #assert(Attacker.check_constraint())

    lambda1_S, impact_S, centrality = Attacker.get_result()
    lambda1_S_0, impact_S_0, centrality_0 = \
            Attacker.lambda1_S_original, Attacker.impact_S_original, Attacker.centrality_original

    lambda1_S_increase_ratio =  (lambda1_S - lambda1_S_0) / lambda1_S_0
    impact_S_increase_ratio = (impact_S - impact_S_0) / torch.abs(impact_S_0)
    centrality_increase_ratio = (centrality - centrality_0) / centrality_0
    utility = Attacker.get_utility()

    return (lambda1_S_increase_ratio.detach().numpy().squeeze(),
            impact_S_increase_ratio.detach().numpy().squeeze(),
            centrality_increase_ratio.detach().numpy().squeeze(),
            utility.detach().numpy().squeeze(),
            addedEdges)





def launch_attach():
    alpha_1, alpha_2, alpha_3 = MAPPING[args.mode]
    Alpha = [alpha_1, alpha_2, alpha_3]
    print("alpha_1: {:.4f}      alpha_2: {:.4f}     alpha_3: {:.4f}\n".format(alpha_1, alpha_2, alpha_3))

    Attacker = Threat_Model(S, S_prime, Alpha, budget_change_ratio, learning_rate, G, args.weighted_graph)
    #Optimizer = torch.optim.Adam(Attacker.parameters(), lr=learning_rate)
    Optimizer = torch.optim.SGD(Attacker.parameters(), lr=learning_rate)

    t1 = time.time()
    lambda1_S_ret, impact_S_ret, centrality_ret, utility_ret, addedEdges = SGD_attack(Attacker, Optimizer)
    print("Time: {:.4f}".format(time.time() - t1))

    avgDegDiff          = Attacker.diff_avgDeg()
    avgDegDiff_S        = Attacker.diff_avgDeg(S)
    avgDegDiff_S_prime  = Attacker.diff_avgDeg(S_prime)
    adjNormDiff         = Attacker.diff_adjNorm()
    addedEdgesRatio     = addedEdges / len(G.edges())

    utility = Attacker.get_utility()

    ## record all the statistics we are interested in
    ret = (utility_ret, lambda1_S_ret, impact_S_ret, centrality_ret, budget_change_ratio, \
        Alpha, avgDegDiff, avgDegDiff_S, avgDegDiff_S_prime, adjNormDiff, addedEdgesRatio, Attacker)
    print("Budget: {:.2f}%  \
           lambda1_S: {:.4f}%  \
           negative impact: {:.4f}% \
           centrality: {:.4f}%  \
           addedEdgesRatio: {:.4f}% \
           utility: {:.4f}\n".format(
              budget_change_ratio*100, lambda1_S_ret*100, impact_S_ret*100, centrality_ret*100, addedEdgesRatio*100, utility_ret))
    print('*' * 80)
    return ret


## the solution from the threat model is a matrix 
## with fractional entries, so we need to round it 
## to a matrix with only integral entries.
def rounding(Attacker):
    A_attacked = Attacker.get_attacked_adj().numpy()
    A = Attacker.original_adj.numpy()

    # A_attacked[A > 0] = 1
    idx = np.nonzero(np.triu(A_attacked, 1) != np.triu(A, 1))
    numChange = len(idx[0])

    modifiedEdges = []
    for i in range(numChange):
        rowIdx, colIdx = idx[0][i], idx[1][i]
        w_a = A_attacked[rowIdx][colIdx]
        w_o = A[rowIdx][colIdx]
        Diff = np.abs(w_a - w_o)
        AddedEdge   = (w_a > w_o and w_o == 0)
        DeletedEdge = (w_a < w_o and w_o >  0)
        if AddedEdge or DeletedEdge:
            modifiedEdges.append((rowIdx, colIdx, Diff, DeletedEdge))
    modifiedEdges = sorted(modifiedEdges, key=lambda x: x[2], reverse=True)

    B = A.copy()
    ## no modification made 
    if len(modifiedEdges) == 0:
        return ( torch.tensor(B, dtype=torch.float32), 0 )

    cnt = 0
    addedEdges   = 0
    deletedEdges = 0
    spec_norm = estimate_sym_specNorm(B-A)
    while (spec_norm <= Attacker.budget):
        Edge = modifiedEdges[cnt]
        DeletedEdge = Edge[-1]
        if not DeletedEdge:
            B[Edge[0], Edge[1]] = 1
            B[Edge[1], Edge[0]] = 1
            addedEdges += 1
        else:
            B[Edge[0], Edge[1]] = 0
            B[Edge[1], Edge[0]] = 0
            deletedEdges += 1  

        ## if budget constraints are violated
        ## revert the operations above
        spec_norm = estimate_sym_specNorm(B-A)
        if spec_norm > Attacker.budget:
            if not DeletedEdge:
                B[Edge[0], Edge[1]] = 0
                B[Edge[1], Edge[0]] = 0
                addedEdges -= 1
            else:
                B[Edge[0], Edge[1]] = 1
                B[Edge[1], Edge[0]] = 1
                deletedEdges -= 1               
            break
        else:     
            cnt += 1
            if cnt >= len(modifiedEdges):
                break
    print("Added edges: {}      Deleted edges: {}\n".format(addedEdges, deletedEdges))
    return ( torch.tensor(B, dtype=torch.float32), addedEdges )


result = defaultdict(list)
graph_ret = defaultdict(list)

#for budget_change_ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
for budget_change_ratio in [0.2]:
    for i in range(args.numExp):
        G = gen_graph(args.graph_type, i)
        mapping = {item: idx for idx, item in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
        adj = nx.adjacency_matrix(G).todense()

        if args.graph_type == "Email":
            S = select_comm(G, True, mapping)
        else:
            S = select_comm(G)

        print("---Comm size: {}    Graph size: {}---".format(len(S), len(G)))

        S_prime = list(set(G.nodes()) - set(S))
        S = torch.LongTensor(S)
        S_prime = torch.LongTensor(S_prime)

        # exhaustively search the best set of hyper-parameters
        opt_sol = launch_attach()
        result[budget_change_ratio].append(opt_sol)

        ## attacked graphs
        Attacker = opt_sol[-1]
        Adj_attacked = Attacker.get_attacked_adj()
        R = contact_matrix(Adj_attacked) 

        G_attacked = nx.from_numpy_matrix(R.numpy(), create_using=nx.Digraph)
        G          = nx.from_numpy_matrix(Attacker.original_adj.numpy()) 

        targets = [True if i in S else False for i in range(G.order())]
        targets = {idx: {'target': targets[idx]} for idx, _ in enumerate(targets)}
        nx.set_node_attributes(G_attacked, targets)
        nx.set_node_attributes(G, targets)
        graph_ret[budget_change_ratio].append({'original': G, 'attacked': G_attacked})


## save attacked graphs to disk
#Key = args.mode
#with open('../result/weighted/min_eigcent_SP/{}_numExp_{}_attacked_graphs_{}.p'.format(args.graph_type, args.numExp, Key), 'wb') as fid:
#    pickle.dump(graph_ret, fid)
#
#
#with open('../result/weighted/min_eigcent_SP/{}_numExp_{}_ret_{}.p'.format(args.graph_type, args.numExp, Key), 'wb') as fid:
#    pickle.dump(result, fid)



