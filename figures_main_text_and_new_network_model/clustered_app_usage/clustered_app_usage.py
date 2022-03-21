import os
import numpy as np
import epipack
from math import exp
#from numpy import random
import random
import networkx as nx
from epipack.stochastic_epi_models import StochasticEpiModel
from scipy.stats import expon
import matplotlib.pyplot as plt
from smallworld import get_smallworld_graph
N = 1000
colors = [
    'dimgrey',
    'lightcoral',
]
parameter = {'number_of_contacts':20}
app = np.linspace(0.001,0.99,10)
def _edge(i,j):
    if i > j:
        return (j,i)
    elif j > i:
        return (i,j)
    else:
        raise ValueError('self-loop')
def get_expon_small_world(N,k0,more_lattice_like=False,node_creation_order='random'):

    G = nx.empty_graph(N)
    degree_seq = [ int(k) for k in expon.rvs(scale=k0,size=N)]
    stubs = list(degree_seq)
    if sum(stubs) % 2 == 1:
        stubs[np.random.randint(0,N-1)] += 1

    if node_creation_order == 'random':
        # generates small world but locally clustered
        order = np.random.permutation(N)
    elif node_creation_order == 'desc':
        # generates locally clustered
        order = np.argsort(stubs)[::-1]
    elif node_creation_order == 'asc':
        # generates locally clustered with short paths
        order = np.argsort(stubs)
    else:
        raise ValueError("`node_creation_order` must be 'random', 'desc', or 'asc', not " + node_creation_order)

    edges = []
    cnt = 0
    for i in order:
        d = 1
        up = True
        while stubs[i] > 0:
            if up:
                j = (i+d) % N
            else:
                j = (i-d) % N
                d += 1
            if i == j:
                break
            if stubs[j] > 0:#and not G.has_edge(i,j):
                edges.append(_edge(int(i),int(j)))
                #G.add_edge(i,j)
                stubs[i] -= 1
                stubs[j] -= 1
            up = not up
            if d >= N//2:
                break
            #f d > N // 2:
            #    print(stubs[i], np.mean(stubs), np.min(stubs),np.max(stubs),cnt)
            #    raise ValueError('Couldn''t find stub')
        cnt += 1
    #print("leftover stubs:",sum(stubs))
    #print("number of nodes with leftover stubs:",np.count_nonzero(stubs))

    #print("len(edges) = ", len(edges), "len(set(edges)) = ", len(set(edges)), "difference = ", len(edges) - len(set(edges)))
    G.add_edges_from(edges)

    return G
def exp_sw_network(N,parameter,**kwargs):
    p = parameter
    k0 = p['number_of_contacts']
    G = get_expon_small_world(N,k0,node_creation_order='asc')
    edge_weight_tuples = [ (e[0], e[1], 1.0) for e in G.edges() ]
    k_norm = 2*len(edge_weight_tuples) / N
    print(k_norm)
    return edge_weight_tuples
def confignetwork(N, parameter,**kwargs):
    p = parameter
    k0 = p['number_of_contacts']
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
    G.remove_edges_from(nx.selfloop_edges(G))
    edge_weight_tuples = [ (e[0], e[1], 1.0) for e in G.edges() ]
    k_norm = 2*len(edge_weight_tuples) / N
    del G
    return edge_weight_tuples
def swnetwork(N, parameter,**kwargs):
    p = parameter
    k_over_2 = int(p['number_of_contacts']/2)
    #beta = 10e-7 #for k = 20, N = 200_000 or k0=10
    beta = 1e-4
    G = get_smallworld_graph(N,k_over_2,beta)
    edge_weight_tuples = [ (e[0], e[1], 1.0) for e in G.edges() ]
    k_norm = 2*len(edge_weight_tuples) / N
    del G
    return edge_weight_tuples
edge_weight_tuples = exp_sw_network(N,parameter)
#edge_weight_tuples = confignetwork(N,parameter)
#edge_weight_tuples = swnetwork(N, parameter)
def get_app_user(N, app):
    if 0 < app < 1:
        mu = N/2
        sigma = 0.03*mu
        def gauss(sigma, mu, x):
            return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2) )
        P = np.array([gauss(sigma, mu, i) for i in range(N)])
        P /= P.sum()
        k = [i for i in range(N)]
        app_user = np.random.choice(a = k,size = int(app*N), p = P, replace = False)
    return list(app_user)

def plot_foundcontacts():
    fig,ax = plt.subplots(1,2)
    app_user = [random.sample(list(range(0, N)), int(N*k)) for k in app]
    counts = [sum(1 for i,j,k in edge_weight_tuples if i in x and j in x) for x in app_user]
    ax[0].plot(app,[i / (len(edge_weight_tuples)) for i in counts], color = colors[0])
    ax[1].hist(app_user[3], color = colors[0], alpha = 0.5,bins = np.arange(0,N+N/10,N/10))
    app_user1 =  [get_app_user(N, k) for k in app]
    counts1 = [sum(1 for i,j,k in edge_weight_tuples if i in x and j in x) for x in app_user1]
    ax[0].plot(app,[i / (len(edge_weight_tuples)) for i in counts1], color = colors[1])
    ax[1].hist(app_user1[3], color = colors[1], alpha = 0.5,bins = np.arange(0,N+N/10,N/10))
    ax[0].set_ylabel("found contacts")
    ax[1].set_ylabel("app participants")
    ax[0].set_xlabel("app participation")
    ax[1].set_xlabel("node index")
    ax[1].set_xlim([0,N])
    plt.show()
plot_foundcontacts()
