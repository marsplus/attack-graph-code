"""
sir_simulations.py
------------------

Simulate SIR dynamics on graph and attacked graph.

"""

import EoN
import pickle
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter

GAMMA = 0.24                     # recovery rate
TAU = 0.06                       # transmission rate
TMAX = 100



def random_bool(size):
    """Return random boolean values."""
    return np.random.randint(2, size=size).astype(bool)


def run_sir(original, attacked, num_sim=500):
    """Sun SIR simulations on both graphs."""
    graphs = {'original': original, 'attacked': attacked}
    rows = []
    for name in graphs:
        G = graphs[name]
        numNode = G.order()
        print('Simulating SIS on {}'.format(name))
        for ns in range(num_sim):
            if ns % 10 == 0: print("numSim: {}".format(ns))
            sim = EoN.fast_SIS(graphs[name], TAU, GAMMA, return_full_data=True)
            # sim = EoN.basic_discrete_SIS(G, TAU/GAMMA, tmax=TMAX, return_full_data=True)

            S = [i for i in range(numNode) if G.nodes[i]['target']]
            SP = list(set(list(range(numNode))) - set(S))

            ## compute the ratio of infected nodes at the end of the epidemic
            inf_ratio_target    = Counter(sim.get_statuses(S, -1).values())['I'] / len(S)
            inf_ratio_bystander = Counter(sim.get_statuses(SP, -1).values())['I'] / len(SP)

            # targets = sum(1 for n in graphs[name]
            #               if graphs[name].nodes[n]['target'] and
            #               sim.node_status(n, -1) == 'I')
            # bystanders = sum(1 for n in graphs[name]
            #                  if not graphs[name].nodes[n]['target'] and
            #                  sim.node_status(n, -1) == 'I')

            rows.append((name, len(sim.I()), emptymax(sim.I()), inf_ratio_target, inf_ratio_bystander))
    return pd.DataFrame(rows, columns=['graph', 't_max', 'i_max',
                                       'ratio targets', 'ratio bystanders'])


def emptymax(arr):
    return arr.max() if arr.shape[0] else 0


# def main():

"""Simulate dynamics on both graphs."""

parser = argparse.ArgumentParser()
parser.add_argument('--graph_type', type=str, default='BA',
                help='graph type')
parser.add_argument('--numExp', type=int, default=1,
                help='numExp')
parser.add_argument('--budget', type=float, default=1,
                help='attacker budget')
args = parser.parse_args()


with open('../result/{}_numExp_{}_attacked_graphs_50quantile.p'.format(args.graph_type, args.numExp), 'rb') as fid:
    graph_ret = pickle.load(fid)

original = graph_ret[args.budget][0]['original']
attacked = graph_ret[args.budget][0]['attacked']

# Run simulations
results = run_sir(original, attacked)

# Display some results
cols = ['ratio targets', 'ratio bystanders']
print(results.pivot_table(index='graph')[cols])



# if __name__ == '__main__':
#     main()
