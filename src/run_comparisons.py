"""
run_comparisons.py
------------------

Run the functions implemented in comparisons.py on our datasets.

"""

import numpy as np
from scipy import sparse
import networkx as nx
from comparisons import centrality_attack, melt_gel


def write_output(graph, filename):
    """NetworkX is bad at writing edgelists, so we do it manually."""
    with open(filename, 'w') as file:
        for u, v, w in graph.edges(data='weight'):
            file.write(f'{u} {v} {w}\n')


def run_all(graph, target, name):
    adj = nx.adjacency_matrix(graph).astype('f')
    eig = sparse.linalg.eigsh(adj, k=1, return_eigenvectors=False)[0]
    for gamma in [0.1, 0.2, 0.3, 0.4, 0.5]:
        print(gamma)
        budget = gamma * eig
        attacked = centrality_attack(graph, target, budget_eig=budget, cent='deg')
        fn = 'results/{}_{}_{}.edges'.format(name, 'deg', gamma)
        write_output(attacked, fn)

        attacked = melt_gel(graph, target, budget_eig=budget)
        fn = 'results/{}_{}_{}.edges'.format(name, 'gel', gamma)
        write_output(attacked, fn)


def read_email():
    graph = nx.read_edgelist('data/datasets/datasets/email/email-Eu-core-cc.txt', nodetype=int)

    # the target is community 37, which should have 16 nodes
    target_nodes = []
    with open('data/datasets/datasets/email/email-Eu-core-department-labels-cc.txt', 'r') as file:
        for line in file:
            node, community = line.split()
            if community == '37':
                target_nodes.append(int(node))
    assert len(target_nodes) == 15
    return graph, graph.subgraph(target_nodes)


def read_brain():
    adj = np.loadtxt('data/datasets/datasets/brain/Brain.txt')
    graph = nx.from_numpy_array(adj)
    target = graph.subgraph(list(range(graph.order() - 100, graph.order())))
    return graph, target


def read_airport():
    graph = nx.Graph()
    with open('data/datasets/datasets/airport/US-airport.txt', 'r') as file:
        for line in file:
            u, v, w = line.split()
            u, v, w = int(u), int(v), float(w)
            if graph.has_edge(u, v):
                graph.edges[u, v]['weight'] += w
                graph.edges[u, v]['count'] += 1
            else:
                graph.add_edge(u, v, weight=w, count=1)

    # take the average
    for u, v in graph.edges():
        graph.edges[u, v]['weight'] /= graph.edges[u, v]['count']

    # normalize between 0 and 1
    max_weight = max(data['weight'] for _, _, data in graph.edges(data=True))
    for u, v in graph.edges():
        graph.edges[u, v]['weight'] /= max_weight

    # target: the node 540 and its neighborhood
    center = 540
    target = graph.subgraph(list(graph.neighbors(center)) + [center])
    assert target.order() == 61

    return graph, target


def main():
    """Load data sets and run baselines."""
    # the README says the last 100 nodes are the target subgraph
    print('brain')
    graph, target = read_brain()
    run_all(graph, target, 'brain')

    print('email')
    graph, target = read_email()
    run_all(graph, target, 'email')

    print('airport')
    graph, target = read_airport()
    run_all(graph, target, 'airport')



if __name__ == '__main__':
    main()
