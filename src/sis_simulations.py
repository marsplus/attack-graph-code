"""
sir_simulations.py
------------------

Simulate SIR dynamics on graph and attacked graph.

"""
import os
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
parser.add_argument('--gamma', type=float, default='0.24',
                help='gamma')
parser.add_argument('--tau', type=float, default='0.2',
                help='tau')
parser.add_argument('--weighted', type=int, default=1,
                    help='whether the graph is weighted')
args = parser.parse_args()


GAMMA = args.gamma                       # recovery rate
TAU = args.tau                           # transmission rate
TMAX = 50
numCPU = 7
LOC = args.location
numSim = 2000
MODE = 'min_eigcent_SP'



def run_sis(original, attacked, budget, num_sim=numSim):
    """Sun SIR simulations on both graphs."""
    graphs = {'original': original, 'attacked': attacked}
    rows = []
    for name in graphs:
        G = graphs[name]
        numNode = G.order()
        print('Simulating SIS on {}'.format(name))
        for ns in range(num_sim):
            if ns % 50 == 0: print("numSim: {}".format(ns))
            #print("numSim: {}".format(ns))
            
            S = [i for i in range(numNode) if G.nodes[i]['target']]
            SP = list(set(range(numNode)) - set(S))

            if name == 'attacked':
                sim = EoN.fast_SIS(graphs[name], TAU, GAMMA, tmax=TMAX, transmission_weight='weight', return_full_data=True)
            else:
                if args.graph_type not in ['Airport', 'Protein', 'Brain']:
                    sim = EoN.fast_SIS(graphs[name], TAU, GAMMA, tmax=TMAX, return_full_data=True)
                else:
                    sim = EoN.fast_SIS(graphs[name], TAU, GAMMA, tmax=TMAX, transmission_weight='weight', return_full_data=True)
            
            ## average results over the last 20 steps
            inf_ratio_target    = np.mean([Counter(sim.get_statuses(S, i).values())['I'] / len(S) for i in range(-1, -21, -1)])
            inf_ratio_bystander = np.mean([Counter(sim.get_statuses(SP, i).values())['I'] / len(SP) for i in range(-1, -21, -1)])
            rows.append((name, inf_ratio_target, inf_ratio_bystander, budget))
    
    return pd.DataFrame(rows, columns=['graph', 'ratio targets', 'ratio bystanders', 'budget'])



def dispatch(params):
    Key = params
    W = 'weighted' if args.weighted else 'unweighted'
    print("Current exp: {}".format(Key))

    with open('../result/{}/{}/{}_numExp_{}_attacked_graphs_{}.p'.format(W, MODE, args.graph_type, args.numExp, Key), 'rb') as fid:
        graph_ret = pickle.load(fid)

    result = []
    for budget in [0.1, 0.2, 0.3, 0.4, 0.5]:
        graph_param = graph_ret[budget]
        for item in graph_param:
            original, attacked = item['original'], item['attacked']
            ret = run_sis(original, attacked, budget)
            result.append(ret)
    result = pd.concat(result)
    return result



"""Simulate dynamics on both graphs."""

pool = Pool(processes=numCPU)
params = []
expName = ['equalAlpha']
for Key in expName:
    params.append(Key)

ret = pool.map(dispatch, params)

W = 'weighted' if args.weighted else 'unweighted'
folder = '../result/{}/{}/{}-SIS/Gamma-{:.2f}---Tau-{:.2f}/'.format(W, MODE, args.graph_type, GAMMA, TAU)
if not os.path.exists(folder):
    os.mkdir(folder)

for idx, Key in enumerate(expName):
    result = ret[idx]
    fileName = '{}_numExp_{}_SIS_{}.p'.format(args.graph_type, args.numExp, Key) 
    with open(os.path.join(folder, fileName), 'wb') as fid:
        pickle.dump(result, fid)

pool.close()
pool.join()
