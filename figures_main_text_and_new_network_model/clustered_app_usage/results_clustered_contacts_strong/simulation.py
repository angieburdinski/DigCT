from smallworld import get_smallworld_graph
import networkx as nx
from scipy.stats import expon
import numpy as np
import random
from math import exp

def get_app_user(N, edge_dict, a):

    if 0 < a < 1:
        app_user = set(random.sample(range(0, N), 1))
        idx_user = set()

        while len(app_user)/N < a:

            idx_user = app_user - idx_user
            all_neighbors = {c for x in idx_user for c in edge_dict[x]}

            if len(app_user)/N < a and len(all_neighbors - app_user) == 0:
                app_user.add(random.choice(list(set(range(0,N)) - app_user)))

            elif len(app_user)/N < a and len(all_neighbors - app_user) > 0:

                while len(app_user)/N < a and len(all_neighbors - app_user) > 0:

                    if len(app_user.union(all_neighbors)) <= a*N:
                        app_user = app_user.union(all_neighbors)

                    elif len(app_user.union(all_neighbors)) > a*N:

                        for idx in idx_user:

                            if len(app_user)/N < a and len(all_neighbors - app_user) > 0:
                                not_app_user = edge_dict[idx] - app_user

                                if len(not_app_user) > 0:
                                    app_user.add(list(not_app_user)[0])
                            else:
                                break


    elif a == 0:
        app_user = set()
    elif a == 1:
        app_user = set(range(0, N))

    return app_user

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
def confignetwork(N, k0,**kwargs):
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
    edge_dict = {idx : {n for n in G.neighbors(idx)} for idx in range(N)}
    return G.edges(), edge_dict
def swnetwork(N, k0,**kwargs):
    k_over_2 = int(k0/2)
    beta = 10e-7
    G = get_smallworld_graph(N,k_over_2,beta)
    edge_dict = {idx : {n for n in G.neighbors(idx)} for idx in range(N)}
    return G.edges(), edge_dict
def ernetwork(N, k0,**kwargs):
    k_over_2 = int(k0/2)
    beta = 1
    G = get_smallworld_graph(N,k_over_2,beta)
    edge_dict = {idx : {n for n in G.neighbors(idx)} for idx in range(N)}
    return G.edges(), edge_dict
def exp_sw_network(N,k0,**kwargs):
    G = get_expon_small_world(N,k0,node_creation_order='asc')
    edge_dict = {idx : {n for n in G.neighbors(idx)} for idx in range(N)}
    return G.edges(), edge_dict
def simulation_code(kwargs):

    N = kwargs['N']
    a = kwargs['a']
    network = kwargs['network']

    if network == 'ER':
        edge_weight_tuples, edge_dict = ernetwork(N, kwargs['number_of_contacts'])
    elif network == 'SW':
        edge_weight_tuples, edge_dict = swnetwork(N, kwargs['number_of_contacts'])
    elif network == 'ER_exp':
        edge_weight_tuples, edge_dict = confignetwork(N, kwargs['number_of_contacts'])
    elif network == 'SW_exp':
        edge_weight_tuples, edge_dict = exp_sw_network(N,kwargs['number_of_contacts'])

    if kwargs['clustered'] == False:
        app = set(random.sample(list(range(0, N)), int(N*a)))
    elif kwargs['clustered'] == True:
        app =  get_app_user(N, edge_dict, a)

    counts = sum(1 for i,j in edge_weight_tuples if i in app and j in app)

    return counts
