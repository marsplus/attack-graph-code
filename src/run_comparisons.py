"""
run_comparisons.py
------------------

Run the functions implemented in comparisons.py on our datasets.

"""

import numpy as np
from scipy import sparse
import networkx as nx
from comparisons import centrality_attack, melt_gel, read_data
from collections import Counter


def write_output(graph, filename):
    """NetworkX is bad at writing edgelists, so we do it manually."""
    with open(filename, 'w') as file:
        for u, v, w in graph.edges(data='weight'):
            file.write(f'{u} {v} {w}\n')


def run_all(graph, target, name, weighted, discount_factor=None):
    adj = nx.adjacency_matrix(graph).astype('f')
    eig = sparse.linalg.eigsh(adj, k=1, return_eigenvectors=False)[0]
    budget = 0.5 * eig

    all_attacked = centrality_attack(
        graph,
        target,
        budget_eig=budget,
        cent='deg',
        weighted=weighted,
        discount_factor=discount_factor,
        milestones=[0.1*eig, 0.2*eig, 0.3*eig, 0.4*eig, 0.5*eig],
    )
    for att, gamma in zip(all_attacked, [0.1, 0.2, 0.3, 0.4, 0.5]):
        fn = 'comparison_results/{}_{}_{}.edges'.format(name, 'deg', gamma)
        zero = Counter(dict(att.degree()).values())[0]
        print(f'There are {zero} nodes with degree zero in {fn}')
        write_output(att, fn)

    # all_attacked = melt_gel(
    #     graph,
    #     target,
    #     budget_eig=budget,
    #     weighted=weighted,
    #     discount_factor=discount_factor,
    #     milestones=[0.1*eig, 0.2*eig, 0.3*eig, 0.4*eig, 0.5*eig],
    # )
    # for att, gamma in zip(all_attacked, [0.1, 0.2, 0.3, 0.4, 0.5]):
    #     fn = 'comparison_results/{}_{}_{}.edges'.format(name, 'gel', gamma)
    #     zero = Counter(dict(att.degree()).values())[0]
    #     print(f'There are {zero} nodes with degree zero in {fn}')
    #     write_output(att, fn)


def main():
    """Load data sets and run baselines."""
    # print('email')
    # graph, target = read_data('Email')
    # run_all(graph, target, 'email', weighted=False)

    print('brain')
    graph, target = read_data('Brain')
    run_all(graph, target, 'brain', weighted=True, discount_factor=0.5)

    # print('airport')
    # graph, target = read_data('Airport')
    # run_all(graph, target, 'airport', weighted=True, discount_factor=0.5)


if __name__ == '__main__':
    main()
