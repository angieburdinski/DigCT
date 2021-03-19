import epipack
import numpy as np
from epipack.stochastic_epi_models import StochasticEpiModel
from math import exp
from numpy import random
import networkx as nx
from smallworld import get_smallworld_graph

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

    edge_weight_tuples = [ (e[0], e[1], 1.0) for e in G.edges() ]

    k_norm = 2*len(edge_weight_tuples) / N
    #print(G.nodes())
    del G
    print(k_norm)
    return edge_weight_tuples, k_norm

def swnetwork(N, **kwargs):
    k_over_2 = 25
    beta = 10e-4
    G = get_smallworld_graph(N,k_over_2,beta)
    edge_weight_tuples = [ (e[0], e[1], 1.0) for e in G.edges() ]
    k_norm = 2*len(edge_weight_tuples) / N
    del G
    print(k_norm)
    return edge_weight_tuples, k_norm

def simulation_code(kwargs):

    def mixed(N, parameter, time, sampling_dt, a, q,**kwargs):

        p = parameter

        #edge_weight_tuples, k_norm = swnetwork(N)
        edge_weight_tuples, k_norm = swnetwork(N)
        kappa = (q*p['recovery_rate'])/(1-q)

        IPa0 = int(random.binomial(p['I_0'], a, 1))
        IP0 = int(p['I_0'] - IPa0)

        Sa0 = int(random.binomial(N-p['I_0'], a, 1))
        S0 = int(N - p['I_0'] - Sa0)

        model = epipack.StochasticEpiModel(['S','E','I_P','I_S','I_A','R','T','X','Sa','Ea','I_Pa','I_Sa','I_Aa','Ra','Ta','Xa','Qa'],N, edge_weight_tuples ,directed=False)

        model.set_conditional_link_transmission_processes({

        ("Ta", "->", "Xa") : [
                ("Xa", "I_Pa", p['z']*p['y'], "Xa", "Ta" ),
                ("Xa", "I_Sa", p['z']*p['y'], "Xa", "Ta" ),
                ("Xa", "I_Aa", p['z']*p['y'], "Xa", "Ta" ),
                ("Xa", "Ea", p['z']*p['y'], "Xa", "Ta" ),
                ("Xa", "Sa", p['z'], "Xa", "Qa" ),
                ("Xa", "I_Pa", p['z']*(1-p['y']), "Xa", "Xa" ),
                ("Xa", "I_Sa", p['z']*(1-p['y']), "Xa", "Xa" ),
                ("Xa", "I_Aa", p['z']*(1-p['y']), "Xa", "Xa" ),
                ("Xa", "Ea", p['z']*(1-p['y']), "Xa", "Xa" )]

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
                    ('Ta',p['chi'],'Xa')])

        model.set_link_transmission_processes([

                    ('I_Pa','S',p['R0']/k_norm*p['beta']/2,'I_Pa','E'),
                    ('I_Aa','S',p['R0']/k_norm*p['recovery_rate']/2,'I_Aa','E'),
                    ('I_Sa','S',p['R0']/k_norm*p['recovery_rate']/2,'I_Sa','E'),

                    ('I_P','Sa',p['R0']/k_norm*p['beta']/2,'I_P','Ea'),
                    ('I_A','Sa',p['R0']/k_norm*p['recovery_rate']/2,'I_A','Ea'),
                    ('I_S','Sa',p['R0']/k_norm*p['recovery_rate']/2,'I_S','Ea'),

                    ('I_Pa','Sa',p['R0']/k_norm*p['beta']/2,'I_Pa','Ea'),
                    ('I_Aa','Sa',p['R0']/k_norm*p['recovery_rate']/2,'I_Aa','Ea'),
                    ('I_Sa','Sa',p['R0']/k_norm*p['recovery_rate']/2,'I_Sa','Ea'),

                    ('I_P','S',p['R0']/k_norm*p['beta']/2,'I_P','E'),
                    ('I_A','S',p['R0']/k_norm*p['recovery_rate']/2,'I_A','E'),
                    ('I_S','S',p['R0']/k_norm*p['recovery_rate']/2,'I_S','E')])

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

        results = max(result['I_S'])+max(result['I_Sa']),max(result['R'])+max(result['Ra'])+max(result['X'])+max(result['Xa'])

        del result

        return results

    results = mixed(**kwargs)

    return results
