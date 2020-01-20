"""
spectrum_attack.py
------------------

Common utilities for performing attack on the Laplacian spectrum.

"""

import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sparse
from scipy.sparse import linalg as splin
import matplotlib.pyplot as plt



mult = np.multiply


def is_symmetric(mat):
    """Check whether a sparse matrix is symmetric."""
    return (mat != mat.T).nnz == 0


def principal_vecs(mat):
    """Get left and right principal eigenvectors of a sparse matrix.

    Params
    ------

    mat (sparse matrix)

    Returns
    -------

    left, right (arrays): the right and left principal eigenvectors.

    """
    if is_symmetric(mat):
        _, vecs = splin.eigs(mat, k=1)
        # left and right eigenvectors of symmetric matrices are equal,
        # and they are always real
        return vecs[:, 0].real, vecs[:, 0].real

    else:
        _, rvecs = splin.eigs(mat, k=1)
        _, lvecs = splin.eigs(mat.T, k=1)
        return lvecs[:, 0], rvecs[:, 0]


def gradient(adj, S, alphas, mask):
    """Gradient (Eq.(5) in the write-up).

    Params
    ------

    adj (sparse matrix): adjacency matrix of the graph.

    S (int sequence): nodes in the subgraph

    alphas (float sequence):

    mask (sparse matrix): mask matrix with a 1 at entry ij whenever nodes i
    and j are neighbors. If the graph is unweighted, this is equal to the
    adjacency matrix.

    """
    I = np.identity(adj.shape[0])

    # The 1st term in the objective function tries to increase the largest
    # eigenvalue of the subgraph S
    adj_sub = get_submat(adj, S)
    left_sub, right_sub = principal_vecs(adj_sub)
    norm_sub = np.dot(left_sub, right_sub)
    outer_sub = np.outer(right_sub, left_sub)
    diag_sub = mult(outer_sub, np.identity(len(S)))
    term1 = np.zeros(adj.shape)
    term1[np.ix_(S, S)] = (outer_sub + outer_sub.T - diag_sub) / norm_sub

    # The 2nd term of the objective function tries not increase the largest
    # eigenvalue of G too much
    left, right = principal_vecs(adj)
    norm = np.dot(left, right)
    outer = np.outer(right, left)
    diag = mult(outer, I)
    term2 = (outer + outer.T - diag) / norm

    # The 3rd term of the objective function tries to increase the
    # centrality of the subgraph S
    indicator_sub = np.zeros(adj.shape[0])
    indicator_sub[S] = 1
    outer_indicator = np.outer(indicator_sub, indicator_sub)
    term3 = 2 * outer_indicator - mult(indicator_sub, I)
    term3 = term3.T
    ones = np.ones((adj.shape[0], adj.shape[0]))
    term3 = np.dot(mult(term3, I), ones) - term3
    term3 = term3 + np.transpose(term3) - mult(term3, I)
    # no need to change the edges' weights within the subgraph
    term3[np.ix_(S, S)] = 0 # why?
    term3 = term3 / len(S)

    # The attacker can only modify existing edges' weights
    term1 = mask.multiply(term1)
    term2 = mask.multiply(term2)
    term3 = mask.multiply(term3)

    return alphas[0] * term1 - alphas[1] * term2 + alphas[2] * term3


def max_eig(mat):
    """Get the largest eigenvalue of a sparse matrix.

    Note: since we are working with symmetric matrices, this function never
    returns a complex number.

    """
    vals = splin.eigs(mat, k=1, return_eigenvectors=False)
    # Sometimes, the adjaecncy matrix of a small subgraph can have negative
    # eigenvalues which are equal in magnitude to the Perron
    # eigenvalue. This may happen for example when the subgraph has three
    # nodes but only one edge. In these cases, it is sufficient to compute
    # the absolute value of the returned eigenvalue. In all other cases,
    # taking the absolute value does not change the result.
    return abs(vals[0].real)


def get_submat(mat, S):
    """Get the submatrix indexed by S.

    Params
    ------

    mat (sparse matrix): the matrix to index.

    S (int sequence): the index sequence.

    """
    indices = np.ix_(S, S)
    try:
        return mat[indices]
    except TypeError as exc:
        return mat.asformat('csr')[indices]


def laplacian(mat):
    """Get the Laplacian corresponding to the adjacency matrix mat."""
    degs = mat.dot(np.ones(mat.shape[1]))
    return sparse.diags(degs) - mat


def subgraph_centrality(lap, x_S):
    """Subgraph centrality of S.

    Params
    ------

    lap (sparse matrix): the graph Laplacian.

    x_S (array): indicator vector of the nodes in subgraph S.

    """
    return x_S.dot(lap.dot(x_S)) / x_S.sum()


def spectrum_attack(graph, subgraph, alphas, learning_rate=0.1,
                    num_iter=100, verbose=False):
    """Run the spectrum attack.

    Change the weights of the existing edges in graph by taking gradient
    descent steps.

    Params
    ------

    graph (nx.Graph): the graph to be attacked. The nodes must be labeled
    by consecutive integers starting at zero.

    subgraph (int sequence): the nodes in the target subgraph.

    alphas (float sequence): the three weights for the three parts of the
    objective function. Their sum must be one.

    learning_rate (float): learning rate for gradient ascent.

    num_iter (int): number of gradient descent steps to perform.

    verbose (bool): whether to print information during computation.

    """
    adj = nx.to_scipy_sparse_matrix(graph).astype(float)
    mask = adj.copy()
    subgraph_indicator = np.zeros(graph.order())
    subgraph_indicator[subgraph] = 1

    rows = []
    for ii in range(num_iter):
        lambda1 = max_eig(adj).real
        lambda1_sub = max_eig(get_submat(adj, subgraph)).real
        lap = laplacian(adj)
        centrality = subgraph_centrality(lap, subgraph_indicator)
        utility = alphas[0] * lambda1_sub - alphas[1] * lambda1 + alphas[2] * centrality
        rows.append([lambda1, lambda1_sub, centrality, utility])
        grad = gradient(adj, subgraph, alphas, mask)
        adj = adj + learning_rate * grad

        if verbose and ii % 10 == 0:
            print('({}) utility: {:.3f}, lambda_1: {:.3f}, '
                  'lambda_1_S: {:.3f}'.format(
                      ii, utility, lambda1, lambda1_sub))

    return pd.DataFrame(rows, columns=['lambda', 'lambda_sub', 'centrality', 'utility']), adj



