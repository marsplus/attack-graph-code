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
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument('--graph_type', type=str, default='BA',
                help='graph type')
parser.add_argument('--numExp', type=int, default=1,
                help='numExp')
parser.add_argument('--budget', type=float, default=1,
                help='attacker budget')
parser.add_argument('--location', type=str, default='random',
                help='locatioin of initial seed')
args = parser.parse_args()


GAMMA = 0.24                     # recovery rate
TAU = 0.06                       # transmission rate
TMAX = 100
numCPU = 7
LOC = args.location


def random_bool(size):
    """Return random boolean values."""
    return np.random.randint(2, size=size).astype(bool)


def run_sis(original, attacked, budget, num_sim=500):
    """Sun SIR simulations on both graphs."""
    graphs = {'original': original, 'attacked': attacked}
    rows = []
    for name in graphs:
        G = graphs[name]
        numNode = G.order()
        print('Simulating SIS on {}'.format(name))
        for ns in range(num_sim):
            if ns % 10 == 0: print("numSim: {}".format(ns))
            
            S = [i for i in range(numNode) if G.nodes[i]['target']]
            SP = list(set(range(numNode)) - set(S))

            ### initially infected nodes appear in S or S_prime
            ### with equal probability
            #if LOC == 'random':
            #    flag = np.random.rand() > 0.5
            #elif LOC == 'S':
            #    flag = True
            #elif LOC == 'SP':
            #    flag = False
            #else:
            #    raise ValueError("Unknown location type\n")

            #if flag:
            #    seed = np.random.choice(S, size=1)
            #else:
            #    seed = np.random.choice(SP, size=1)

            sim = EoN.fast_SIS(graphs[name], TAU, GAMMA, tmax=TMAX, return_full_data=True)

            ## compute the ratio of infected nodes at the end of the epidemic
            inf_ratio_target    = Counter(sim.get_statuses(S, -1).values())['I'] / len(S)
            inf_ratio_bystander = Counter(sim.get_statuses(SP, -1).values())['I'] / len(SP)

            rows.append((name, inf_ratio_target, inf_ratio_bystander, budget))
    return pd.DataFrame(rows, columns=['graph', 'ratio targets', 'ratio bystanders', 'budget'])


def emptymax(arr):
    return arr.max() if arr.shape[0] else 0



def dispatch(params):
    graph_param, budget = params[0], params[1]
    original, attacked = graph_param['original'], graph_param['attacked']
    results = run_sis(original, attacked, budget)
    return results



"""Simulate dynamics on both graphs."""

pool = Pool(processes=numCPU)


for Key in ['equalAlpha']:
    with open('../result/utility_max/{}_numExp_{}_attacked_graphs_{}.p'.format(args.graph_type, args.numExp, Key), 'rb') as fid:
        graph_ret = pickle.load(fid)

    result = []
    for budget in [0.1, 0.2, 0.3, 0.4, 0.5]:
        params = zip(graph_ret[budget], [budget]*args.numExp)
        ret = pool.map(dispatch, params)
        result.extend(ret)
    result = pd.concat(result)

    fidName = '../result/utility_max/{}-SIS/{}_numExp_{}_SIS_{}.p'.format(args.graph_type, args.graph_type, args.numExp, Key) 
    with open(fidName, 'wb') as fid:
        pickle.dump(result, fid)
