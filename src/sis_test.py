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
import matplotlib.pyplot as plt

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


GAMMA = 0.24                      # recovery rate
TAU = 0.06                        # transmission rate
TMAX = 100
numCPU = 7
LOC = args.location
numSim = 6

def random_bool(size):
    """Return random boolean values."""
    return np.random.randint(2, size=size).astype(bool)


def run_sis(original, attacked, budget, num_sim=numSim):
    """Sun SIR simulations on both graphs."""
    graphs = {'original': original, 'attacked': attacked}
    rows = {i+1: {'original': None, 'attacked': None} for i in range(num_sim)}
    for name in graphs:
        G = graphs[name]
        numNode = G.order()
        print('Simulating SIS on {}'.format(name))
        for ns in range(num_sim):
            if ns % 50 == 0: print("numSim: {}".format(ns))
            #print("numSim: {}".format(ns))
            
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
            
            numSteps = len(sim.t())
            inf_ratio_target    = sum(sim.summary(S)[1]['I']) / (len(S) * numSteps)
            inf_ratio_bystander = sum(sim.summary(SP)[1]['I']) / (len(SP) * numSteps)

            ### compute the ratio of infected nodes at the end of the epidemic
            #inf_ratio_target    = Counter(sim.get_statuses(S, -1).values())['I'] / len(S)
            #inf_ratio_bystander = Counter(sim.get_statuses(SP, -1).values())['I'] / len(SP)
    
            rows[ns+1][name] = sim
    return rows



def dispatch(params):
    Key = params
    print("Current exp: {}".format(Key))

    with open('../result/utility_max/min_eigcent_SP/{}_numExp_{}_attacked_graphs_{}.p'.format(args.graph_type, args.numExp, Key), 'rb') as fid:
        graph_ret = pickle.load(fid)
    
    budget = 0.4
    graph_param = graph_ret[budget]
    item = graph_param[0]
    original, attacked = item['original'], item['attacked']
    ret = run_sis(original, attacked, budget)
    return ret

    #result = []
    #for budget in [0.1, 0.2, 0.3, 0.4, 0.5]:
    #    graph_param = graph_ret[budget]
    #    for item in graph_param:
    #        original, attacked = item['original'], item['attacked']
    #        ret = run_sis(original, attacked, budget)
    #        result.append(ret)
    #result = pd.concat(result)
    #return result



"""Simulate dynamics on both graphs."""

pool = Pool(processes=numCPU)
params = []
#expName = ['alpha1=1', 'alpha2=0', 'alpha3=0', 'alpha3=1', 'equalAlpha']
expName = ['equalAlpha']
for Key in expName:
    params.append(Key)

ret = dispatch(params[0])

#ret = pool.map(dispatch, params)

#for idx, Key in enumerate(expName):
#    result = ret[idx]
#    fidName = '../result/utility_max/min_eigcent_SP/{}-SIS-new/Gamma-{:.2}---Tau-{:.2f}/{}_numExp_{}_SIS_{}.p'.format(args.graph_type, GAMMA, TAU, args.graph_type, args.numExp, Key) 
#    with open(fidName, 'wb') as fid:
#        pickle.dump(result, fid)

pool.close()
pool.join()
