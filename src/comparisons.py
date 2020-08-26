"""
comparisons.py
--------------

Run experiments against baselines and competitor algorithms.

"""

import numpy as np
import networkx as nx
import scipy.sparse as sparse


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


def degree_centrality(graph, e):
    """Return the sum of the degrees at the endpoints."""
    return graph.degree(e[0]) + graph.degree(e[1])


cent_func = {'deg': degree_centrality}


def max_cent_edge(graph, cent='deg'):
    """Return the edge with the highest centrality."""
    cur_cent = float('-inf')
    cur_edge = None
    for e in graph.edges():
        if cent_func[cent](graph, e) > cur_cent:
            cur_edge = e
    return cur_edge


def centrality_attack(graph, target, budget_edges=None, budget_eig=None, cent='deg'):
    """Use edge centrality to attack the graph spectrum.

    This function removes edges of high centrality from outside the target
    subgraph, and adds edges of high centrality to the target subgraph.

    Parameters
    ----------
    graph (nx.Graph): the graph to attack

    target (nx.GraphView): the target subgraph.  Note this must be a view,
    i.e. it must update itself automatically whenever graph is modified

    budget_edges (int): the number of edges to modify

    budget_eig (float): the desired amount to change the graph spectrum

    cent (str): the edge centrality to use.  Possible values are 'deg', which
    uses the sum of the degrees at the endpoints of the edge;

    Notes
    -----
    Only one of budget_edges, budget_eig can be different than None.

    This function alternates between adding and removing, until the budget is
    spent.  The first step is always to add an edge to the target subgraph.

    Returns
    -------
    A nx.Graph object that represents the graph after the attack

    """
    if budget_edges and budget_eig:
        raise ValueError('budget_edges and budget_eig cannot both be non null')
    if budget_edges is None and budget_eig is None:
        raise ValueError('budget_edges and budget_eig cannot both be None')

    adj = nx.adjacency_matrix(graph)
    attacked = graph.copy()
    target = attacked.subgraph([n for n in target])
    outside_target = attacked.subgraph([n for n in attacked if n not in target])

    spent_budget = 0
    if budget_edges:
        keep_going = lambda sb: sb <= budget_edges
    if budget_eig:
        keep_going = lambda sb: sb <= budget_eig

    mode = 'add'
    failure_prev = False
    while True:
        # choose which edge to add/remove
        if mode == 'add':
            edge = max_cent_edge(nx.complement(target), cent=cent)
        else:
            edge = max_cent_edge(outside_target, cent=cent)

        # if we fail twice in a row, stop
        if not edge:
            if failure_prev:
                print(f'No more edges to add or remove. Stopping. Budget spent: {spent_budget:.3f}')
                break
            else:
                print('Did not find edge to add/remove.')
                failure_prev = True
                continue

        # check whether applying the changes would keep us within budget
        if budget_edges:
            spent_budget += 1
        if budget_eig:
            # there is no implementation for 2-norms for spectral matrices
            # so we must convert to dense...
            staged_changes = attacked.copy()
            if mode == 'add':
                staged_changes.add_edge(*edge)
            else:
                staged_changes.remove_edge(*edge)
            staged_adj = nx.adjacency_matrix(staged_changes)
            diff = (adj - staged_adj).A
            spent_budget = np.linalg.norm(diff, ord=2)
        if not keep_going(spent_budget):
            print(f'Applying the next change would incur in {spent_budget:.3f} budget. Stopping.')
            break

        # If we reach here, we can finally apply the changes (and not just
        # stage them).  Note we cannot just assign attacked = staged_changes
        # because attacked needs to reamain a SubGraphView of the original
        # graph.
        if mode == 'add':
            print(f'add: ({edge[0]:02d}, {edge[1]:02d}).', end='')
            attacked.add_edge(*edge)
        else:
            print(f'rem: ({edge[0]:02d}, {edge[1]:02d}).', end='')
            attacked.remove_edge(*edge)

        print(f'\tTotal budget spent: {spent_budget:.3f}')

        # swtich between adding and removing
        mode = 'rem' if mode == 'add' else 'add'

    return attacked


def main():
    """Run experiments."""
    graph = nx.barabasi_albert_graph(100, 3)

    random_node = np.random.choice([n for n in graph])
    target = graph.subgraph(graph.neighbors(random_node))
    budget = 2

    attacked = centrality_attack(graph, target, budget_eig=budget, cent='deg')

    # don't forget to add more centrality measures
    # melt_gel(graph, target, budget_eig=budget)

    # then ask Sixie for his datasets to run experiments or give the code to
    # Sixie to run the experiments
    # Datasets are at http://tonghanghang.org/events/netrin/netrin_data.tgz



if __name__ == '__main__':
    main()
