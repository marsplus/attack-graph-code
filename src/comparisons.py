"""
comparisons.py
--------------

Run experiments against baselines and competitor algorithms.

"""

import random
import numpy as np
import networkx as nx
import scipy.sparse as sparse


def left_right(matrix):
    """Return the principal left and right eigenvectors of the matrix."""
    right = sparse.linalg.eigs(matrix, k=1, return_eigenvectors=True)[1]
    right = right.reshape((-1,)).real
    left = sparse.linalg.eigs(matrix.T, k=1, return_eigenvectors=True)[1]
    left = left.reshape((-1,)).real
    return left, right


def melt(graph, k):
    """See Algorithm 1 in [1].

    Parameters
    ----------

    graph (nx.Graph): the graph to melt

    k (int): the number of edges to remove from the graph

    Notes
    -----
    If k < graph.size(), then the edges will be returned in the order of their
    gel score.  However, if k >= graph.size(), then all of the edges of the
    graph are returned in arbitrary order.

    References
    ----------
    [1] Hanghang Tong, B. Aditya Prakash, Tina Eliassi-Rad, Michalis Faloutsos,
    Christos Faloutsos: Gelling, and melting, large graphs by edge
    manipulation. CIKM 2012: 245-254

    """
    n, m = graph.order(), graph.size()
    # Can't remove edges from the empty graph
    if m == 0:
        return None

    # Can't remove more edges than there are in the graph
    if k >= graph.size():
        return list(graph.edges())

    # The numbered comments correspond exactly to the number lines of Algorithm
    # 2 in Reference [1].  They have been only slightly modified for clarity
    A = nx.adjacency_matrix(graph).astype('f')

    # 1: compute the leading eigenvalue λ of A; let u and v be the
    # corresponding left and right eigenvectors, respectively
    u, v = left_right(A)

    # 2: if mini=1,...,nu(i) < 0 then
    # 3: assign u ← −u
    # 4: end if
    if u.min() < 0:
        u *= -1

    # 5: if mini=1,...,nv(i) < 0 then
    # 6: assign v ← −v
    # 7: end if
    if v.min() < 0:
        v *= -1

    # 8: for each edge e=(i, j) with A[i, j]=1 do
    # 9: score(e) = u[i] * v[j]
    # 10: end for
    score = {}
    for i, j in zip(*A.nonzero()):
        if i != j and (i, j) not in score and (j, i) not in score:
            score[(i, j)] = u[i] * v[j]

    # 11: return top-k edges with the highest score
    edges = sorted(score, key=score.get)[-k:]
    nodes = list(graph.nodes)
    return [(nodes[u], nodes[v]) for u, v in edges]


def gel(graph, k):
    """See Algorithm 2 in [1].

    Parameters
    ----------

    graph (nx.Graph): the graph to gel

    k (int): the number of edges to add to the graph

    References
    ----------
    [1] Hanghang Tong, B. Aditya Prakash, Tina Eliassi-Rad, Michalis Faloutsos,
    Christos Faloutsos: Gelling, and melting, large graphs by edge
    manipulation. CIKM 2012: 245-254

    """
    n, m = graph.order(), graph.size()
    # Can't add edges to the complete graph
    if m == n*(n-1)/2:
        return None

    # Can't add more edges than are missing
    if k >= n*(n-1)/2 - m:
        return [(u, v) for u in graph for v in graph if (u, v) not in graph.edges]

    # The numbered comments correspond exactly to the number lines of Algorithm
    # 2 in Reference [1].  They have been only slightly modified for clarity

    # 1: compute the left (u) and right (v) eigenvectors of A that correspond
    # to the leading eigenvalue (u, v ≥ 0)
    A = nx.adjacency_matrix(graph).astype('f')
    u, v = left_right(A)
    if u.max() < 0:
        u *= -1
    if v.max() < 0:
        v *= -1

    # 2: calculate the maximum in-degree (d_in) and out-degree (d_out) of A
    d_in, d_out = int(A.sum(axis=0).max()), int(A.sum(axis=1).max())

    # 3: find the subset of k + d_in nodes with the highest left eigenscores
    # u_i. Index them by I
    num_edges = min(k + d_in, graph.size()) # do not take more edges than exist
    I = np.argsort(u)#[num_edges:]

    # 4: find the subset of k + d_out nodes with the highest right eigenscores
    # v_j. Index them by J
    num_edges = min(k + d_out, graph.size()) # do not take more edges than exist
    J = np.argsort(u)#[num_edges:]

    # 5. index by P the set of all edges e=(i,j), i∈I, j∈J with A(i,j)=0
    P = [(i, j) for i in I for j in J
         if abs(A[i, j]) < 1e-5  # add only if they are not already neighbors
         and i != j              # don't add self-loops
    ]

    # 6: for each in P, define score(e) := u(i) * v(j)
    score = {}
    for i, j in P:
        if i != j and (i, j) not in score and (j, i) not in score:
            score[(i, j)] = u[i] * v[j]

    # 8: return top-k non-existing edges with the highest scores among P.
    edges = sorted(score, key=score.get)[-k:]
    nodes = list(graph.nodes)
    return [(nodes[u], nodes[v]) for u, v in edges]


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
    ###
    ### Setup
    ###

    # Check that we only have one type of budget
    if budget_edges and budget_eig:
        raise ValueError('budget_edges and budget_eig cannot both be non null')
    if budget_edges is None and budget_eig is None:
        raise ValueError('budget_edges and budget_eig cannot both be None')

    # If the budget is given in eigenvalue units, we must modify one edge at a
    # time, for which we use the specialized function centrality_attack
    if budget_eig:
        return centrality_attack(graph, target, budget_edges=None,
                                 budget_eig=budget_eig, cent='gel')

    # Compute the necessary graphs.  Make sure they are SubGraphViews of
    # attacked, not of the original graph.
    attacked = graph.copy()
    target = attacked.subgraph(list(target.nodes()))
    outside_target = attacked.subgraph([n for n in attacked
                                        if n not in target])

    # We split the budget proportionally to the size of the target and original
    # graphs.  For example, if the target has 5% of the edges of the original
    # graph, then 5% of the budget will be used to add edges to the target
    # subgraph, and 95% will be used to remove edges from outside of the target
    # subgraph.  Remember, at this point we are only using budget_edges.
    fraction = target.size() / graph.size()
    target_budget = int(fraction * budget_edges)
    outside_budget = budget_edges - target_budget

    ###
    ### Main function
    ###

    to_add = gel(target, target_budget)
    to_rem = melt(outside_target, outside_budget)
    attacked.add_edges_from(to_add)
    attacked.remove_edges_from(to_rem)
    return attacked


def max_absent_edge(graph, cent='deg'):
    """Return the absent edge with the maximum centrality.

    Parameters
    ----------

    graph (nx.Graph): the graph to analyze

    cent (str): one of 'deg' or 'bet'

    Notes
    -----
    If cent='bet', the centrality of an absent edge is defined as the sum of
    the betweenness centrality of the two nodes at its endpoints.

    """
    absent_edges = [(u, v)
                    for u in graph
                    for v in graph
                    if (u, v) not in graph.edges
                    and u != v]
    if cent == 'deg':
        deg = graph.degree()
        cur_max = float('-inf')
        cur_edge = None
        for e in absent_edges:
            if deg[e[0]] + deg[e[1]] > cur_max:
                cur_edge = e
        return cur_edge

    elif cent == 'bet':
        bet = nx.betweenness_centrality(graph)
        cent_dict = {e: bet(e[0]) + bet(e[1]) for e in absent_edges}
        return max(cent_dict, key=cent_dict.get) if cent_dict else None


def max_edge(graph, cent='deg'):
    """Return the edge with the maximum centrality.

    Parameters
    ----------

    graph (nx.Graph): the graph to analyze

    cent (str): one of 'deg' or 'bet'

    Notes
    -----
    If cent='bet', this uses nx.edge_betweenness_centrality

    """
    if cent == 'deg':
        deg = graph.degree()
        cur_max = float('-inf')
        cur_edge = None
        for e in graph.edges():
            if deg[e[0]] + deg[e[1]] > cur_max:
                cur_edge = e
        return cur_edge
    elif cent == 'bet':
        cent_dict = nx.edge_betweenness_centrality(graph)
        return max(cent_dict, key=cent_dict.get) if cent_dict else None


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
    uses the sum of the degrees at the endpoints of the edge; 'bet', which uses
    edge betweenness centrality; and 'gel', which uses the NetGel algorithm.

    Notes
    -----
    Only one of budget_edges, budget_eig can be different than None.

    This function alternates between adding and removing, until the budget is
    spent.  The first step is always to add an edge to the target subgraph.

    Returns
    -------
    A nx.Graph object that represents the graph after the attack

    """
    ###
    ### Setup
    ###

    # Do not use this function with cent='gel' and budget_edges not None
    assert not (cent == 'gel' and budget_edges), 'Use gel_melt() instead'

    # Check that we only have one type of budget
    if budget_edges and budget_eig:
        raise ValueError('budget_edges and budget_eig cannot both be non null')
    if budget_edges is None and budget_eig is None:
        raise ValueError('budget_edges and budget_eig cannot both be None')

    # Compute the necessary graphs.  Make sure they are SubGraphViews of
    # attacked, not of the original graph.
    adj = nx.adjacency_matrix(graph)
    attacked = graph.copy()
    target = attacked.subgraph(list(target.nodes()))
    outside_target = attacked.subgraph([n for n in attacked
                                        if n not in target])

    # The keep_going function makes sure we compare to the correct budget
    spent_budget = 0
    if budget_edges:
        keep_going = lambda sb: sb <= budget_edges
    if budget_eig:
        keep_going = lambda sb: sb <= budget_eig

    # Toggle between adding ('add') and removing ('rem') edges
    mode = 'add'

    # get_edge_to_add must return a non-existing edge outside of the target
    # subgraph that will be added.  get_edge_to_rem must return an existing
    # edge inside of the target subgraph that will be removed.  Most of the
    # time we will choose them like this...
    get_edge_to_add = lambda: max_absent_edge(target, cent=cent)
    get_edge_to_rem = lambda: max_edge(outside_target, cent=cent)
    # ... except when the centality is 'gel'
    if cent == 'gel':
        if budget_eig:
            # These return the first returned edge or an empty list
            get_edge_to_add = lambda: (gel(target, 1) or [[]])[0]
            get_edge_to_rem = lambda: (melt(outside_target, 1) or [[]])[0]
        else:
            print('If you want to use NetGel with a budget given '
                  'by a number of edges, use melt_gel() instead.')
            return


    ###
    ### Main loop
    ###

    # If we fail to find an edge to mofidy twice in a row, we will break
    failure_prev = False

    while True:

        # Choose which edge to add/remove
        edge = get_edge_to_add() if mode == 'add' else get_edge_to_rem()

        # If we fail twice in a row, stop
        if not edge:
            if failure_prev:
                print(f'No more edges to add or remove. Stopping. Budget spent: {spent_budget:.3f}')
                break
            else:
                print('Did not find edge to add/remove.')
                failure_prev = True
                mode = 'rem' if mode == 'add' else 'add'
                continue

        # Check whether applying the changes would keep us within budget
        if budget_edges:
            spent_budget += 1
        if budget_eig:
            staged_changes = attacked.copy()
            if mode == 'add':
                staged_changes.add_edge(*edge)
            else:
                staged_changes.remove_edge(*edge)
            staged_adj = nx.adjacency_matrix(staged_changes)
            # there is no implementation for 2-norms for spectral matrices
            # so we must convert to dense...
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

        # Toggle between adding and removing
        mode = 'rem' if mode == 'add' else 'add'

    return attacked


def main():
    """Run experiments."""
    # graph = nx.barabasi_albert_graph(500, 3)
    # for _, _, data in graph.edges(data=True):
    #     data['weight'] = random.randint(0, 10)
    graph = nx.karate_club_graph()

    # make sure to use graph.subgraph to get a SubGraphView
    random_nodes = np.random.choice(graph, size=10)
    target = graph.subgraph(
        [neigh for rn in random_nodes for neigh in graph.neighbors(rn)]
        + [rn for rn in random_nodes]
    )
    budget = 20

    attacked = centrality_attack(graph, target, budget_eig=budget, cent='deg')
    attacked = melt_gel(graph, target, budget_eig=budget)



if __name__ == '__main__':
    main()
