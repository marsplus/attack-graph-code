"""
graph_generators.py
-------------------

Generate random graphs.

"""

import os
import tempfile
import subprocess
import numpy as np
import networkx as nx
from time import time


def hyperbolic_graph(size, avgk, gamma):
    # Tested with hyperbolic_graph_generator v1.0.3:
    # https://github.com/named-data/Hyperbolic-Graph-Generator/releases
    command = 'hyperbolic_graph_generator'
    size_flag = '-n{}'.format(size)
    avgk_flag = '-k{}'.format(avgk)
    gamma_flag = '-g{}'.format(gamma)
    seed_flag = '-s{}'.format(time())
    quiet_flag = '-q'
    _, tmp = tempfile.mkstemp()
    dir_flag = '-o{}'.format(tempfile.gettempdir())
    file_flag = '-f{}'.format(os.path.split(tmp)[-1])
    subprocess.run([command, dir_flag, file_flag, size_flag, avgk_flag,
                    gamma_flag, seed_flag, quiet_flag],
                   stdout=subprocess.DEVNULL)
    graph = read_hyperbolic_graph(tmp + '.hg', size)
    os.remove(tmp)
    assert graph.order() == size
    return nx.convert_node_labels_to_integers(graph)


def read_hyperbolic_graph(filename, size):
    """Reads a hyperbolic file *.hg into a networkx.Graph object."""
    # The first line of the file is information that we do not care
    # about. The following size lines contain the position in hyperbolic
    # space of each node. The following lines contain the edges.
    with open(filename) as graph_file:
        graph_file.readline()
        graph = nx.Graph()
        for _ in range(size):
            node, xcoord, ycoord = graph_file.readline().split()
            graph.add_node(int(node), xcoord=float(xcoord), ycoord=float(ycoord))
        for line in graph_file:
            graph.add_edge(*[int(x) for x in line.split()])
        graph.graph['name'] = 'hyperbolic(N={})'.format(size)
    return graph


def kronecker_graph(size, density, initiator=None):
    """Generate a stochastic Kronecker graph using SNAP."""
    command = 'krongen'
    tmp = tempfile.NamedTemporaryFile()
    output_flag = '-o:{}'.format(tmp.name)
    iters = int(round(np.log2(size)))
    iter_flag = '-i:{}'.format(iters)
    seed_flag = '-s:{}'.format(time())

    # The expected number of edges is (\sum_ij a_ij)^k, where a_ij is the
    # matrix passed in mat_flag, and k is the number of iterations.
    mat = initiator if initiator is not None else initiator_matrix(size, density)
    mat = '; '.join([', '.join([str(el) for el in row]) for row in mat])
    mat_flag = '-m:{}'.format(mat)
    subprocess.run([command, output_flag, iter_flag, mat_flag, seed_flag],
                   stdout=subprocess.DEVNULL)
    graph = nx.read_edgelist(tmp.name, nodetype=int)
    graph.graph['name'] = 'Kronecker(N={}, m={})'.format(2**iters, mat)

    # assert graph.order() == size
    return nx.convert_node_labels_to_integers(graph)


def initiator_matrix(size, density):
    """Return an initiator matrix for the stochastic Kronecker model.

    The stochastic Kronecker random graph needs an initiator matrix. This
    function returns a 2x2 initiator matrix from which the stochastic
    Kronecker model will generate a network with expected number of nodes
    and edges equal to size and density, respectively.

    With an initiator matrix [a, b; b, c], the expected number of nodes and
    edges is 2**k, and (a+2b+c)**k, respectively, where k is the number of
    iterations of the Kronecker product. This allows us to draw a, b, and c
    randomly in such a way that fixes the expected density.

    """
    iters = int(round(np.log2(size)))
    # We have to make sure that the resulting initiator matrix generates a
    # densification power law.  Thus, we draw a, and c randomly until the
    # condition for densification is satisfied. (See
    # https://arxiv.org/pdf/0812.4905.pdf, section 3.7 'Densification'.)
    # We also fix a around 0.95 and c near 0.48, as inspired by the values
    # of Table 4 from the reference.
    mina, maxa = 0.90, 0.99
    minc, maxc = 0.46, 0.51
    a, b, c = 0, 0, 0
    while a + 2*b + c < 2:
        a = (maxa - mina) * np.random.random() + mina
        c = (maxc - minc) * np.random.random() + minc
        b = (np.power(density * size * (size-1) / 2, 1/iters) - a - c) / 2
    # np.array([[0.9, 0.46], [0.43, 0.48]])
    return [[a, b], [b, c]]
