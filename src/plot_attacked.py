"""
plot_attacked.py
----------------

Plot the weights of attacked graphs.

"""

import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from comparisons import read_data, norm


def read_attacked(fn, nodes, weighted=False):
    attacked = nx.Graph()
    attacked.add_nodes_from(nodes)
    with open(fn) as file:
        for line in file:
            if weighted:
                node1, node2, weight = line.split()
                attacked.add_edge(int(node1), int(node2), weight=float(weight))
            else:
                node1, node2, _ = line.split()
                attacked.add_edge(int(node1), int(node2))
    return attacked


def main(name, cent):
    """Load attacked graph and visualize the weights."""
    graph, target = read_data(name)
    name = name.lower()
    outside_target = graph.subgraph([n for n in graph if n not in target])
    adj = nx.adjacency_matrix(graph).astype('f')
    eig = sparse.linalg.eigsh(adj, k=1, return_eigenvectors=False)[0]
    attacked = read_attacked(f'comparison_results/{name}_{cent}_0.4.edges', graph.nodes(), weighted=True)
    attacked_target = attacked.subgraph([n for n in target])
    attacked_outside_target = attacked.subgraph([n for n in attacked if n not in attacked_target])

    used = norm(adj - nx.adjacency_matrix(attacked))
    print(f'budget used: {used:.3f}, total budget: {eig:.3f}')

    _, axes = plt.subplots(1, 2, figsize=(9, 4))

    edges_bef = sorted(target.edges, key=lambda e: target.edges[e]['weight'])
    edges_aft = sorted(attacked_target.edges, key=lambda e: attacked_target.edges[e]['weight'])
    axes[0].set_title('Changes in target subgraph')
    axes[0].plot([target.edges[e]['weight'] for e in edges_bef], label='original')
    axes[0].plot([attacked_target.edges[e]['weight'] for e in edges_aft], label='attacked')
    axes[0].set_xlabel('Edges in the target subgraph, sorted by weight')
    axes[0].set_ylabel('weight')
    axes[0].legend()

    edges_bef = sorted(outside_target.edges, key=lambda e: outside_target.edges[e]['weight'])
    edges_aft = sorted(attacked_outside_target.edges, key=lambda e: attacked_outside_target.edges[e]['weight'])
    axes[1].set_title('Changes outside of target subgraph')
    axes[1].plot([outside_target.edges[e]['weight'] for e in edges_bef], label='original')
    axes[1].plot([attacked_outside_target.edges[e]['weight'] for e in edges_aft], label='attacked')
    axes[1].set_xlabel('Edges outside the target subgraph, sorted by weight')
    axes[1].set_ylabel('weight')
    axes[1].legend()

    plt.suptitle('Brain')
    plt.show()
    # plt.savefig(f'/tmp/{name}_{cent}_plot.png', dpi=144, bbox_inches='tight')



if __name__ == '__main__':
    main()
