"""
run_comparisons.py
------------------

Run the functions implemented in comparisons.py on our datasets.

"""

import numpy as np
from scipy import sparse
import networkx as nx
from comparisons import centrality_attack, melt_gel, read_email, read_brain, read_airport
from collections import Counter


def write_output(graph, filename):
    """NetworkX is bad at writing edgelists, so we do it manually."""
    with open(filename, 'w') as file:
        for u, v, w in graph.edges(data='weight'):
            file.write(f'{u} {v} {w}\n')


def run_all(graph, target, name, weighted, discount_factor=None):
    adj = nx.adjacency_matrix(graph).astype('f')
    eig = sparse.linalg.eigsh(adj, k=1, return_eigenvectors=False)[0]
    for gamma in [0.1, 0.2, 0.3, 0.4, 0.5]:
        print(gamma)
        budget = gamma * eig
        attacked = centrality_attack(
            graph,
            target,
            budget_eig=budget,
            cent='deg',
            weighted=weighted,
            discount_factor=discount_factor
        )
        fn = 'comparison_results/{}_{}_{}.edges'.format(name, 'deg', gamma)
        zero = Counter(dict(attacked.degree()).values())[0]
        print(f'There are {zero} nodes with degree zero in {fn}')
        write_output(attacked, fn)

        attacked = melt_gel(
            graph,
            target,
            budget_eig=budget,
            weighted=weighted,
            discount_factor=discount_factor
        )
        fn = 'comparison_results/{}_{}_{}.edges'.format(name, 'gel', gamma)
        zero = Counter(dict(attacked.degree()).values())[0]
        print(f'There are {zero} nodes with degree zero in {fn}')
        write_output(attacked, fn)


def main():
    """Load data sets and run baselines."""
    print('email')
    graph, target = read_email()
    run_all(graph, target, 'email', weighted=False)

    print('brain')
    graph, target = read_brain()
    run_all(graph, target, 'brain', weighted=True, discount_factor=0.5)

    print('airport')
    graph, target = read_airport()
    run_all(graph, target, 'airport', weighted=True, discount_factor=0.5)



if __name__ == '__main__':
    main()
