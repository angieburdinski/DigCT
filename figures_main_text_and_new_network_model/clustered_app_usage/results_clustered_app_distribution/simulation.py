import epipack
import numpy as np
from epipack.stochastic_epi_models import StochasticEpiModel
from math import exp
import networkx as nx
import random
from smallworld import get_smallworld_graph
from scipy.stats import expon

"""
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
    return app_user
"""
def get_app_user(N, edge_dict, a):

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
    edge_weight_tuples = [ (e[0], e[1], 1.0) for e in G.edges() ]
    k_norm = 2*len(edge_weight_tuples) / N
    return edge_weight_tuples,edge_dict,k_norm
def swnetwork(N, k0,**kwargs):
    k_over_2 = int(k0/2)
    beta = 10e-7
    G = get_smallworld_graph(N,k_over_2,beta)
    edge_dict = {idx : {n for n in G.neighbors(idx)} for idx in range(N)}
    edge_weight_tuples = [ (e[0], e[1], 1.0) for e in G.edges() ]
    k_norm = 2*len(edge_weight_tuples) / N
    return edge_weight_tuples,edge_dict,k_norm
def ernetwork(N, k0,**kwargs):
    k_over_2 = int(k0/2)
    beta = 1
    G = get_smallworld_graph(N,k_over_2,beta)
    edge_dict = {idx : {n for n in G.neighbors(idx)} for idx in range(N)}
    edge_weight_tuples = [ (e[0], e[1], 1.0) for e in G.edges() ]
    k_norm = 2*len(edge_weight_tuples) / N
    return edge_weight_tuples,edge_dict,k_norm
def exp_sw_network(N,k0,**kwargs):
    G = get_expon_small_world(N,k0,node_creation_order='asc')
    edge_dict = {idx : {n for n in G.neighbors(idx)} for idx in range(N)}
    edge_weight_tuples = [ (e[0], e[1], 1.0) for e in G.edges() ]
    k_norm = 2*len(edge_weight_tuples) / N
    return edge_weight_tuples,edge_dict,k_norm
def simulation_code(kwargs):

    def mixed(networks, a, clustered, N, parameter, sampling_dt, time, **kwargs):

        p = parameter
        kappa = (p["q"]*p["recovery_rate"])/(1-p["q"])

        if networks == 'ER':
            edge_weight_tuples, edge_dict, k_norm = ernetwork(N, p['number_of_contacts'])
        if networks == 'SW':
            edge_weight_tuples, edge_dict, k_norm = swnetwork(N, p['number_of_contacts'])
        if networks == 'ER_exp':
            edge_weight_tuples, edge_dict, k_norm = confignetwork(N, p['number_of_contacts'])
        if networks == 'SW_exp':
            edge_weight_tuples, edge_dict, k_norm = exp_sw_network(N,p['number_of_contacts'])


        model = epipack.StochasticEpiModel(['S','E','I_P','I_S','I_A','R','T','X',\
                                            'Sa','Ea','I_Pa','I_Sa','I_Aa','Ra','Ta','Xa','Qa','C'],\
                                            N, edge_weight_tuples, directed=False)

        model.set_conditional_link_transmission_processes({
        ("Ta", "->", "Xa") : [
                ("Xa", "I_Pa", p["y"], "Xa", "Ta" ),
                ("Xa", "I_Sa", p["y"], "Xa", "Ta" ),
                ("Xa", "I_Aa", p["y"], "Xa", "Ta" ),
                ("Xa", "Ea", p["y"], "Xa", "Ta" ),
                ("Xa", "Sa", "->", "Xa", "Qa" ),
                ("Xa", "I_Pa", (1-p["y"]), "Xa", "C" ),
                ("Xa", "I_Sa", (1-p["y"]), "Xa", "C" ),
                ("Xa", "I_Aa", (1-p["y"]), "Xa", "C" ),
                ("Xa", "Ea", (1-p["y"]), "Xa", "C" )]
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
                    ('Ta',(1-p["z"])*p['chi'],'X')
                    ])

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
                    ('I_S','S',p["R0"]/k_norm*p['recovery_rate']/2,'I_S','E')
                    ])

        model.set_network(N, edge_weight_tuples)

        id_S = model.get_compartment_id("S")
        id_Sa = model.get_compartment_id("Sa")
        id_I_P = model.get_compartment_id("I_P")
        id_I_Pa = model.get_compartment_id("I_Pa")

        if clustered == True:
            if 0 < a < 1:
                app_user = get_app_user(N,edge_dict, a)
                not_app_user = [i for i in list(range(N)) if i not in app_user]

                I_Pa_list = list(np.random.choice(list(app_user), int(a*p['I_0']), replace=False))
                Sa_list = [i for i in app_user if i not in I_Pa_list]
                I_P_list = list(np.random.choice(not_app_user, int(p['I_0']-(a*p['I_0'])), replace=False))
                S_list = [i for i in not_app_user if i not in I_P_list]

                initial_node_statuses = np.zeros(N,dtype=int)

                for x in range(N):
                    if x in S_list:
                        initial_node_statuses[x] = id_S
                    elif x in Sa_list:
                        initial_node_statuses[x] = id_Sa
                    elif x in I_P_list:
                        initial_node_statuses[x] = id_I_P
                    elif x in I_Pa_list:
                        initial_node_statuses[x] = id_I_Pa

                model.set_node_statuses(initial_node_statuses)

            elif a == 0 or a == 1:
                IPa0 = int(np.random.binomial(p['I_0'], a, 1))
                IP0 = int(p['I_0'] - IPa0)
                Sa0 = int(np.random.binomial(N-p['I_0'], a, 1))
                S0 = int(N - p['I_0'] - Sa0)
                x = model.set_random_initial_conditions({ 'Sa': Sa0, 'S': S0, 'I_P': IP0, 'I_Pa': IPa0})
                I_Pa_list = [i for i,v in enumerate(x.node_status) if v == id_I_Pa]
                Sa_list = [i for i,v in enumerate(x.node_status) if v == id_Sa]

        else:
            IPa0 = int(np.random.binomial(p['I_0'], a, 1))
            IP0 = int(p['I_0'] - IPa0)
            Sa0 = int(np.random.binomial(N-p['I_0'], a, 1))
            S0 = int(N - p['I_0'] - Sa0)
            x = model.set_random_initial_conditions({ 'Sa': Sa0, 'S': S0, 'I_P': IP0, 'I_Pa': IPa0})

            I_Pa_list = [i for i,v in enumerate(x.node_status) if v == id_I_Pa]
            Sa_list = [i for i,v in enumerate(x.node_status) if v == id_Sa]

        t, result = model.simulate(tmax = time , sampling_dt = sampling_dt)
        return max(result['R']),max(result['Ra']),max(result['X']),max(result['Xa']),max(result['C'])

    results = mixed(**kwargs)

    return results
