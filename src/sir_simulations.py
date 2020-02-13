"""
sir_simulations.py
------------------

Simulate SIR dynamics on graph and attacked graph.

"""

import EoN
import numpy as np
import pandas as pd
import networkx as nx

GAMMA = 1.0                     # recovery rate
TAU = 5                         # transmission rate



def random_bool(size):
    """Return random boolean values."""
    return np.random.randint(2, size=size).astype(bool)


def run_sir(original, attacked, num_sim=100):
    """Sun SIR simulations on both graphs."""
    graphs = {'original': original, 'attacked': attacked}
    rows = []
    for name in graphs:
        print('Simulating SIR on {}'.format(name))
        for _ in range(num_sim):
            sim = EoN.fast_SIR(graphs[name], TAU, GAMMA, return_full_data=True)
            targets = sum(1 for n in graphs[name]
                          if graphs[name].nodes[n]['target'] and
                          sim.node_status(n, -1) == 'R')
            bystanders = sum(1 for n in graphs[name]
                             if not graphs[name].nodes[n]['target'] and
                             sim.node_status(n, -1) == 'R')
            rows.append((name, len(sim.I()), emptymax(sim.I()),
                         emptymax(sim.R()), targets, bystanders))
    return pd.DataFrame(rows, columns=['graph', 't_max', 'i_max', 'r_max',
                                       'infected_targets', 'infected_bystanders'])


def emptymax(arr):
    return arr.max() if arr.shape[0] else 0


def main():
    """Simulate dynamics on both graphs."""
    # For this example, I will use the Karate Club graph, choose a target
    # subgraph at random, and use a dummy version of the attacked graph
    original = nx.karate_club_graph()
    targets = random_bool(original.order())
    targets = {n: targets[idx] for idx, n in enumerate(original)}
    print('There are {} targets'.format(len([n for n in targets if targets[n]])))
    nx.set_node_attributes(original, targets, 'target')

    # In actuality, the attacked graph should be the output of our attack
    # algorithms
    attacked = original.copy()
    attacked.remove_nodes_from([33, 2])

    # Run simulations
    results = run_sir(original, attacked)

    # Display some results
    cols = ['infected_targets', 'infected_bystanders']
    print(results.pivot_table(index='graph')[cols])



if __name__ == '__main__':
    main()
