import pickle
import numpy as np
import networkx as nx


adj = np.loadtxt('protein_adj_mat.txt', delimiter='\t')
G = nx.from_numpy_array(adj)
comps = list(nx.connected_components(G))
CC = max(comps, key=lambda x: len(x))
deleted_nodes = set(G.nodes()) - CC

G = G.subgraph(CC)
mapping = {item: idx for idx, item in enumerate(G.nodes())}
G = nx.relabel_nodes(G, mapping)


nodeID_to_prot = {}
with open('nodeID_to_protein.txt', 'r') as fid:
    for line in fid:
        (key, value) = line.split()
        nodeID_to_prot[int(key)] = value

prot_to_nodeID = {value: key for key, value in nodeID_to_prot.items()}

deleted_prot = [nodeID_to_prot[i] for i in deleted_nodes]
imm_prot = set(['O09528', 'O12160', 'O12161', 'P03407', 'P04324', 'P04325', 'P04326', 'P04604', 'P05888', 'P69700', 'P69728', 'Q85737', 'Q9QPN3'])
inter = set(deleted_prot).intersection(imm_prot)

for item in inter:
    imm_prot.remove(item)


S = [mapping[prot_to_nodeID[prot]] for prot in imm_prot]
print(S)

with open('protein_network.p', 'wb') as fid:
    pickle.dump(G, fid)
