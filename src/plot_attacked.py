"""
plot_attacked.py
----------------

Plot the weights of attacked graphs.

"""

import networkx as nx
import matplotlib.pyplot as plt
from comparisons import read_email, read_brain, read_airport


def read_attacked(fn, weight=False):
    return nx.read_edgelist(
        fn,
        delimiter=' ',
        nodetype=int,
        data=([('weight', float)] if weight else False))


def main():
    """Load attacked graph and visualize the weights."""
    graph, target = read_brain()
    outside_target = graph.subgraph([n for n in graph if n not in target])
    attacked = read_attacked('comparison_results/brain_deg_0.4.edges', weight=True)
    attacked_target = attacked.subgraph(list(range(attacked.order() - 100, attacked.order())))
    attacked_outside_target = attacked.subgraph([n for n in attacked if n not in target])

    edges_bef = sorted(target.edges, key=lambda e: target.edges[e]['weight'])
    edges_aft = sorted(target.edges, key=lambda e: attacked_target.edges[e]['weight'])
    plt.title('Changes in target subgraph')
    plt.plot([target.edges[e]['weight'] for e in edges_bef], label='original')
    plt.plot([attacked_target.edges[e]['weight'] for e in edges_aft], label='attacked')
    plt.legend()
    plt.show()

    edges_bef = sorted(outside_target.edges, key=lambda e: outside_target.edges[e]['weight'])
    edges_aft = sorted(outside_target.edges, key=lambda e: attacked_outside_target.edges[e]['weight'])
    plt.title('Changes outside of target subgraph')
    plt.plot([outside_target.edges[e]['weight'] for e in edges_bef], label='original')
    plt.plot([attacked_outside_target.edges[e]['weight'] for e in edges_aft], label='attacked')
    plt.legend()
    plt.show()



    is it only adding _new_ edges?




if __name__ == '__main__':
    main()
