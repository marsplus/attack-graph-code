"""
comparisons.py
--------------

Run experiments against baselines and competitor algorithms.

"""

import networkx as nx



def melt_gel(graph, target, budget_edges=None, budget_eig=None):
    """Use NetMelt and NetGel to attack the graph spectrum.

    The NetMelt algorithm removes edges in order to decrease the largest
    eigenvalue the most.  The NetGel algorithm adds new edges in order to
    increase the largest eigenvalue the most.  See reference [1].

    This function applies NetMelt outside the target subgraph, and NetGel
    inside the target subgraph.

    Parameters
    ----------
    graph (nx.Graph): the graph to attack

    target (nx.Graph): the target subgraph

    budget_edges (int): the number of edges to modify.  budget_edges / 2 edges
    are added to the target subgraph, while budget_edges / 2 edges are removed
    from outside of it

    budget_eig (float): the desired amount to change the graph spectrum

    Notes
    -----
    Only one of budget_edges, budget_eig can be different than None

    References
    ----------
    [1] Hanghang Tong, B. Aditya Prakash, Tina Eliassi-Rad, Michalis Faloutsos,
    Christos Faloutsos: Gelling, and melting, large graphs by edge
    manipulation. CIKM 2012: 245-254

    """
    # BTW, the NetMelt code in Matlab among with some competitors is at
    # http://tonghanghang.org/events/netrin/netrin_edge.tgz
    pass

def centrality_attack(graph, target, budget_edges=None, budget_eig=None, cent='deg'):
    """Use edge centrality to attack the graph spectrum.

    This function removes edges of high centrality from outside the target
    subgraph, and adds edges of high centrality to the target subgraph.

    Parameters
    ----------
    graph (nx.Graph): the graph to attack

    target (nx.Graph): the target subgraph

    budget_edges (int): the number of edges to modify.  budget_edges / 2 edges
    are added to the target subgraph, while budget_edges / 2 edges are removed
    from outside of it

    budget_eig (float): the desired amount to change the graph spectrum

    cent (str): the edge centrality to use.  Possible values are 'deg', which
    uses the sum of the degrees at the endpoints of the edge;

    Notes
    -----
    Only one of budget_edges, budget_eig can be different than None

    """
    # Here all we need to do is choose the centrality measure
    pass


def main():
    """Run experiments."""
    graph = nx.karate_club_graph()
    target = some_subgraph_here()
    budget = 2

    centrality_attack(graph, target, budget_eig=budget):
    melt_gel(graph, target, budget_eig=budget)

    # then ask Sixie for his datasets to run experiments or give the code to
    # Sixie to run the experiments
    # Datasets are at http://tonghanghang.org/events/netrin/netrin_data.tgz


if __name__ == '__main__':
    main()
