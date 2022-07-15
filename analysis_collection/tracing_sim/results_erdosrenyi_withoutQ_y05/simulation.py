import epipack
import numpy as np
from epipack.stochastic_epi_models import StochasticEpiModel
from math import exp
from numpy import random
import networkx as nx
from smallworld import get_smallworld_graph
from scipy.stats import expon
import numpy as np
import networkx as nx

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
    return edge_weight_tuples, k_norm
def swnetwork(N, parameter,**kwargs):
    p = parameter
    k_over_2 = int(p['number_of_contacts']/2)
    #beta = 10e-7 #for k = 20, N = 200_000 or k0=10
    beta = 1
    G = get_smallworld_graph(N,k_over_2,beta)
    edge_weight_tuples = [ (e[0], e[1], 1.0) for e in G.edges() ]
    k_norm = 2*len(edge_weight_tuples) / N
    del G
    return edge_weight_tuples, k_norm
def exp_sw_network(N,parameter,**kwargs):
    p = parameter
    k0 = p['number_of_contacts']
    G = get_expon_small_world(N,k0,node_creation_order='random')
    edge_weight_tuples = [ (e[0], e[1], 1.0) for e in G.edges() ]
    k_norm = 2*len(edge_weight_tuples) / N
    del G
    print(k_norm)
    return edge_weight_tuples, k_norm
def simulation_code(kwargs):

    def mixed(N, parameter, time, sampling_dt,quarantiningS, a, q, y, **kwargs):
        p = parameter
        #edge_weight_tuples, k_norm = confignetwork(N,parameter)
        edge_weight_tuples, k_norm = swnetwork(N, parameter)
        kappa = (q*p['recovery_rate'])/(1-q)
        IPa0 = int(random.binomial(p['I_0'], a, 1))
        IP0 = int(p['I_0'] - IPa0)
        Sa0 = int(random.binomial(N-p['I_0'], a, 1))
        S0 = int(N - p['I_0'] - Sa0)
        if quarantiningS == True:
            model = epipack.StochasticEpiModel(['S','E','I_P','I_S','I_A','R','T','X','Sa','Ea','I_Pa','I_Sa','I_Aa','Ra','Ta','Xa','Qa','C'],N, edge_weight_tuples ,directed=False)
            model.set_conditional_link_transmission_processes({
            ("Ta", "->", "Xa") : [
                    ("Xa", "I_Pa", y, "Xa", "Ta" ),
                    ("Xa", "I_Sa", y, "Xa", "Ta" ),
                    ("Xa", "I_Aa", y, "Xa", "Ta" ),
                    ("Xa", "Ea", y, "Xa", "Ta" ),
                    ("Xa", "Sa", "->", "Xa", "Qa" ),
                    ("Xa", "I_Pa", (1-y), "Xa", "C" ),
                    ("Xa", "I_Sa", (1-y), "Xa", "C" ),
                    ("Xa", "I_Aa", (1-y), "Xa", "C" ),
                    ("Xa", "Ea", (1-y), "Xa", "C" )]
                    })
            model.set_node_transition_processes([
                        ('E',p['alpha'],'I_P'),
                        ('I_P',(1-p['x'])*p['beta'],'I_S'),
                        ('I_P',p['x']*p['beta'],'I_A'),
                        ('I_A',p['recovery_rate'],'R'),
                        ('I_S',p['recovery_rate'],'R'),
                        ('I_S',kappa,'T'),
                        ('T',p['chi'],'X'),
                        ('Qa',p['omega'],'Sa'),
                        ('Ea',p['alpha'],'I_Pa'),
                        ('I_Pa',(1-p['x'])*p['beta'],'I_Sa'),
                        ('I_Pa',p['x']*p['beta'],'I_Aa'),
                        ('I_Aa',p['recovery_rate'],'Ra'),
                        ('I_Sa',p['recovery_rate'],'Ra'),
                        ('I_Sa',kappa,'Ta'),
                        ('Ta',p["z"]*p['chi'],'Xa'),
                        ('Ta',(1-p["z"])*p['chi'],'X')])

        elif quarantiningS == False:
            model = epipack.StochasticEpiModel(['S','E','I_P','I_S','I_A','R','T','X','Sa','Ea','I_Pa','I_Sa','I_Aa','Ra','Ta','Xa','C'],N, edge_weight_tuples ,directed=False)
            model.set_conditional_link_transmission_processes({
            ("Ta", "->", "Xa") : [
                    ("Xa", "I_Pa", y, "Xa", "Ta" ),
                    ("Xa", "I_Sa", y, "Xa", "Ta" ),
                    ("Xa", "I_Aa", y, "Xa", "Ta" ),
                    ("Xa", "Ea", y, "Xa", "Ta" ),
                    ("Xa", "I_Pa", (1-y), "Xa", "C" ),
                    ("Xa", "I_Sa", (1-y), "Xa", "C" ),
                    ("Xa", "I_Aa", (1-y), "Xa", "C" ),
                    ("Xa", "Ea", (1-y), "Xa", "C" )]
                    })
            model.set_node_transition_processes([
                        ('E',p['alpha'],'I_P'),
                        ('I_P',(1-p['x'])*p['beta'],'I_S'),
                        ('I_P',p['x']*p['beta'],'I_A'),
                        ('I_A',p['recovery_rate'],'R'),
                        ('I_S',p['recovery_rate'],'R'),
                        ('I_S',kappa,'T'),
                        ('T',p['chi'],'X'),
                        ('Ea',p['alpha'],'I_Pa'),
                        ('I_Pa',(1-p['x'])*p['beta'],'I_Sa'),
                        ('I_Pa',p['x']*p['beta'],'I_Aa'),
                        ('I_Aa',p['recovery_rate'],'Ra'),
                        ('I_Sa',p['recovery_rate'],'Ra'),
                        ('I_Sa',kappa,'Ta'),
                        ('Ta',p["z"]*p['chi'],'Xa'),
                        ('Ta',(1-p["z"])*p['chi'],'X')])
        model.set_link_transmission_processes([

                    ('I_Pa','S',p["R0"]/k_norm*p['beta']/2,'I_Pa','E'),
                    ('I_Aa','S',p["R0"]/k_norm*p['recovery_rate']/2,'I_Aa','E'),
                    ('I_Sa','S',p["R0"]/k_norm*p['recovery_rate']/2,'I_Sa','E'),

                    ('I_P','Sa',p["R0"]/k_norm*p['beta']/2,'I_P','Ea'),
                    ('I_A','Sa',p["R0"]/k_norm*p['recovery_rate']/2,'I_A','Ea'),
                    ('I_S','Sa',p["R0"]/k_norm*p['recovery_rate']/2,'I_S','Ea'),

                    ('I_Pa','Sa',p["R0"]/k_norm*p['beta']/2,'I_Pa','Ea'),
                    ('I_Aa','Sa',p["R0"]/k_norm*p['recovery_rate']/2,'I_Aa','Ea'),
                    ('I_Sa','Sa',p["R0"]/k_norm*p['recovery_rate']/2,'I_Sa','Ea'),

                    ('I_P','S',p["R0"]/k_norm*p['beta']/2,'I_P','E'),
                    ('I_A','S',p["R0"]/k_norm*p['recovery_rate']/2,'I_A','E'),
                    ('I_S','S',p["R0"]/k_norm*p['recovery_rate']/2,'I_S','E')])
        model.set_network(N, edge_weight_tuples)

        del edge_weight_tuples

        model.set_random_initial_conditions({ 'Sa': Sa0, 'S': S0, 'I_P': IP0, 'I_Pa': IPa0})

        del p
        del a
        del q
        del N

        t, result = model.simulate(tmax = time , sampling_dt = sampling_dt)

        del model
        del t
        del time
        del sampling_dt

        results = max(result['R']),max(result['Ra']),max(result['X']),max(result['Xa']),max(result['C'])

        del result

        return results
    results = mixed(**kwargs)

    return results
