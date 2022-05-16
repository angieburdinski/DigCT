from smallworld import get_smallworld_graph
import networkx as nx
from scipy.stats import expon
import numpy as np
import random
from math import exp
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
    G.add_edges_from(edges)

    return G
def exp_sw_network(N, number_of_contacts,**kwargs):
    G = get_expon_small_world(N,k0 = number_of_contacts,node_creation_order='asc')
    edge_weight_tuples = [ (e[0], e[1], 1.0) for e in G.edges() ]
    del G
    return edge_weight_tuples
def sw_network(N, number_of_contacts,**kwargs):

    k_over_2 = number_of_contacts/2
    beta = 1e-6
    G = get_smallworld_graph(N,k_over_2,beta)
    edge_weight_tuples = [ (e[0], e[1], 1.0) for e in G.edges() ]
    del G
    return edge_weight_tuples

def simulation_code(kwargs):
    N = kwargs['N']
    a = kwargs['a']

    if kwargs['network'] == 'sw':
        edge_weight_tuples = sw_network(N, kwargs['number_of_contacts'])
    else:
        edge_weight_tuples = exp_sw_network(N,kwargs['number_of_contacts'])

    if kwargs['clustered'] == True:
        app = random.sample(list(range(0, N)), int(N*a))
        counts = sum(1 for i,j,k in edge_weight_tuples if i in app and j in app)
    else:
        app =  get_app_user(N, a)
        counts = sum(1 for i,j,k in edge_weight_tuples if i in app and j in app)

    del edge_weight_tuples
    return app, counts
