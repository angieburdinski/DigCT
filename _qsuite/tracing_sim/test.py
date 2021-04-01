import epipack
import numpy as np
from epipack.stochastic_epi_models import StochasticEpiModel
from math import exp
from numpy import random
import networkx as nx
from smallworld import get_smallworld_graph
from SamplableSet import SamplableSet
import matplotlib.pyplot as plt

def swnetwork(N, **kwargs):
    k_over_2 = 10
    beta = 10e-7
    G = get_smallworld_graph(N,k_over_2,beta)
    print(nx.average_clustering(G))
    print(len(G.edges())/N)
    print(nx.average_shortest_path_length(G))
def confignetwork(N,**kwargs):
    k0 = 20
    def expodegree(x):
        return 1/k0*exp(-x/k0)

    P = []
    k_i = []
    for i in range(N-1):
        p_k = expodegree(i)
        P.append(p_k)
        k_i.append(i)
    P = np.array(P)
    P /= P.sum()

    def seq(k_i,P):
        expected_degree_sequence = np.linspace(0,1,2)
        while sum(expected_degree_sequence) % 2 != 0:
            expected_degree_sequence = np.random.choice(
              k_i,
              N,
              p = P
            )

        return expected_degree_sequence

    expected_degree_sequence = seq(k_i,P)

    G = nx.configuration_model(expected_degree_sequence,create_using = nx.Graph())
    #G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    print(nx.average_clustering(G))
    print(len(G.edges())/N)
    print(nx.average_shortest_path_length(G))
def expodegree(x):
    return 1/k0*exp(-x/k0)
#k0 = 20
#x = [i for i in range(200000)]
#y = [expodegree(i) for i in range(200000)]
#plt.plot(x,y,color = 'k',ls = 'solid')
#plt.yscale('log')
#plt.xscale('log')
#plt.show()
print(np.linspace(0,1,25))
