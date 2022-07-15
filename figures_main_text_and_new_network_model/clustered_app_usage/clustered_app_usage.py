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
import matplotlib.ticker as mtick
from smallworld import get_smallworld_graph
import time
lss = ['solid','dashed']
colors = ['#333333','#888888']
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
    edge_dict = {idx : {n for n in G.neighbors(idx)} for idx in range(N)}
    return  G.edges(), edge_dict
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
    edge_dict = {idx : {n for n in G.neighbors(idx)} for idx in range(N)}
    return G.edges(),edge_dict
def erdos_network(N, parameter,**kwargs):
    p = parameter
    k_over_2 = int(p['number_of_contacts']/2)
    beta = 1
    G = get_smallworld_graph(N,k_over_2,beta)
    edge_dict = {idx : {n for n in G.neighbors(idx)} for idx in range(N)}
    return G.edges(), edge_dict
def swnetwork(N, parameter,**kwargs):
    p = parameter
    k_over_2 = int(p['number_of_contacts']/2)
    if N == 1000:
        beta = 1e-3
    if N == 10000:
        beta = 1e-5
    G = get_smallworld_graph(N,k_over_2,beta)
    edge_dict = {idx : {n for n in G.neighbors(idx)} for idx in range(N)}
    return G.edges(), edge_dict
def lattice_network(N, parameter,**kwargs):
    p = parameter
    k_over_2 = int(p['number_of_contacts']/2)
    beta = 0
    G = get_smallworld_graph(N,k_over_2,beta)
    edge_dict = {idx : {n for n in G.neighbors(idx)} for idx in range(N)}
    return G.edges(), edge_dict
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
def get_app_user_dense(N, edge_dict, a):

    if 0 < a < 1:
        app_user = set(random.sample(range(0, N), int(0.01*N)))
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
def get_app_user_sparse(N, edge_dict, app):

    if 0 < app < 1:
        a = 0.3*app + app
        app_user = set(random.sample(range(0, N), int(0.01*N)))
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

        while len(app_user)/N > app:
            app_user.pop()

    elif app == 0:
        app_user = set()
    elif app == 1:
        app_user = set(range(0, N))

    return app_user

def found_contacts():
    N =  10_000
    parameter = {'number_of_contacts':20}
    app = np.linspace(0,1,11)

    network_list = ['ER','SW','ER_exp','SW_exp']
    #network_list = ['SW','SW_exp']
    fig,ax = plt.subplots(1,len(network_list))
    its = 10
    for i_n, network in enumerate(network_list):

        ax[i_n].set_xlabel("app participation",loc = "right")
        ax[i_n].spines['top'].set_visible(False)
        ax[i_n].spines['right'].set_visible(False)
        ax[i_n].xaxis.set_major_formatter(mtick.PercentFormatter())

        non_clus_counts = np.zeros((its,len(app)))
        clus_counts = np.zeros((its,len(app)))
        if network == 'ER':
            edge_weight_tuples, edge_dict = erdos_network(N, parameter)
        if network == 'SW':
            edge_weight_tuples, edge_dict = swnetwork(N, parameter)
        if network == 'ER_exp':
            edge_weight_tuples, edge_dict = confignetwork(N, parameter)
        if network == 'SW_exp':
            edge_weight_tuples, edge_dict = exp_sw_network(N,parameter)

        for it in range(its):
            non_clus_app = [set(random.sample(range(0,N), int(N*k))) for k in app]
            clus_app =  [get_app_user_sparse(N, edge_dict,k) for k in app]
            non_clus_counts[it] = np.array([sum(1 for i,j in edge_weight_tuples if i in x and j in x) for x in non_clus_app])
            clus_counts[it] = np.array([sum(1 for i,j in edge_weight_tuples if i in x and j in x) for x in clus_app])


        ax[i_n].plot(app * 100,[i/len(edge_weight_tuples) for i in np.mean(non_clus_counts,axis = 0)],label = 'random app participation', color = colors[1], ls = lss[1], marker = 'o')
        ax[i_n].plot(app * 100,[i/len(edge_weight_tuples) for i in np.mean(clus_counts,axis = 0)], label = 'clustered app participation', color = colors[0], ls = lss[0], marker = 'o')

    ax[0].legend()
    ax[0].set_ylabel("found contacts",loc = "bottom")
    plt.show()
def found_contacts_hist():
    N =  1_000
    parameter = {'number_of_contacts':20}
    app = 0.3
    network_list = ['ER','SW','ER_exp','SW_exp']
    its = 100
    for i_n, network in enumerate(network_list):


        plt.xticks([],color='w')
        #plt.spines['top'].set_visible(False)
        #plt.spines['right'].set_visible(False)
        #ax[i_n].xaxis.set_major_formatter(mtick.PercentFormatter())

        non_clus_counts = np.zeros((its,1))
        clus_counts = np.zeros((its,1))
        if network == 'ER':
            edge_weight_tuples, edge_dict = erdos_network(N, parameter)
        if network == 'SW':
            edge_weight_tuples, edge_dict = swnetwork(N, parameter)
        if network == 'ER_exp':
            edge_weight_tuples, edge_dict = confignetwork(N, parameter)
        if network == 'SW_exp':
            edge_weight_tuples, edge_dict = exp_sw_network(N,parameter)

        for it in range(its):
            non_clus_app = set(random.sample(range(0,N), int(N*app)))
            clus_app =  get_app_user_sparse(N, edge_dict,app)
            non_clus_counts[it] = np.array(sum(1 for i,j in edge_weight_tuples if i in non_clus_app and j in non_clus_app))
            clus_counts[it] = np.array(sum(1 for i,j in edge_weight_tuples if i in clus_app and j in clus_app))

        plt.bar(i_n-0.2,np.mean(non_clus_counts,axis = 0)/len(edge_weight_tuples))
        plt.bar(i_n+0.2,np.mean(clus_counts,axis = 0)/len(edge_weight_tuples))

    #ax[0].legend()
    #ax[0].set_ylabel("found contacts",loc = "bottom")
    plt.show()
found_contacts_hist()
