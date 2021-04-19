import epipack
import numpy as np
from epipack.stochastic_epi_models import StochasticEpiModel
from math import exp
from numpy import random
import networkx as nx
from smallworld import get_smallworld_graph
import matplotlib.pyplot as plt
N = 200_000
q = 0.3
a =  0.3
R0 = 2.5
z = 0.64
parameter = {
        'chi':1/2.5,
        'recovery_rate' : 1/7,
        'alpha' : 1/3,
        'beta' : 1/2,
        'number_of_contacts' : 20,
        'x':0.17,
        'y':0.1,
        'I_0' : N*0.01,
        'omega':1/10
        }

sampling_dt = 0.5
time = 10e6

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
    #G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))

    edge_weight_tuples = [ (e[0], e[1], 1.0) for e in G.edges() ]

    k_norm = 2*len(edge_weight_tuples) / N
    #print(G.nodes())
    del G
    print(k_norm)
    return edge_weight_tuples, k_norm


def mixed(N, parameter, time, sampling_dt, a, q,z, R0, **kwargs):

    p = parameter

    edge_weight_tuples, k_norm = confignetwork(N,parameter)
    #edge_weight_tuples, k_norm = swnetwork(N,parameter)
    kappa = (q*p['recovery_rate'])/(1-q)

    IPa0 = int(random.binomial(p['I_0'], a, 1))
    IP0 = int(p['I_0'] - IPa0)

    Sa0 = int(random.binomial(N-p['I_0'], a, 1))
    S0 = int(N - p['I_0'] - Sa0)

    model = epipack.StochasticEpiModel(['S','E','I_P','I_S','I_A','R','T','X','Sa','Ea','I_Pa','I_Sa','I_Aa','Ra','Ta','Xa','Qa','C'],N, edge_weight_tuples ,directed=False)

    model.set_conditional_link_transmission_processes({

    ("Ta", "->", "Xa") : [
            ("Xa", "I_Pa", z*p['y'], "Xa", "Ta" ),
            ("Xa", "I_Sa", z*p['y'], "Xa", "Ta" ),
            ("Xa", "I_Aa", z*p['y'], "Xa", "Ta" ),
            ("Xa", "Ea", z*p['y'], "Xa", "Ta" ),
            ("Xa", "Sa", z, "Xa", "Qa" ),
            ("Xa", "I_Pa", z*(1-p['y']), "Xa", "C" ),
            ("Xa", "I_Sa", z*(1-p['y']), "Xa", "C" ),
            ("Xa", "I_Aa", z*(1-p['y']), "Xa", "C" ),
            ("Xa", "Ea", z*(1-p['y']), "Xa", "C" )]

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

                ('I_Pa','S',R0/k_norm*p['beta']/2,'I_Pa','E'),
                ('I_Aa','S',R0/k_norm*p['recovery_rate']/2,'I_Aa','E'),
                ('I_Sa','S',R0/k_norm*p['recovery_rate']/2,'I_Sa','E'),

                ('I_P','Sa',R0/k_norm*p['beta']/2,'I_P','Ea'),
                ('I_A','Sa',R0/k_norm*p['recovery_rate']/2,'I_A','Ea'),
                ('I_S','Sa',R0/k_norm*p['recovery_rate']/2,'I_S','Ea'),

                ('I_Pa','Sa',R0/k_norm*p['beta']/2,'I_Pa','Ea'),
                ('I_Aa','Sa',R0/k_norm*p['recovery_rate']/2,'I_Aa','Ea'),
                ('I_Sa','Sa',R0/k_norm*p['recovery_rate']/2,'I_Sa','Ea'),

                ('I_P','S',R0/k_norm*p['beta']/2,'I_P','E'),
                ('I_A','S',R0/k_norm*p['recovery_rate']/2,'I_A','E'),
                ('I_S','S',R0/k_norm*p['recovery_rate']/2,'I_S','E')])

    model.set_network(N, edge_weight_tuples)

    del edge_weight_tuples

    model.set_random_initial_conditions({ 'Sa': Sa0, 'S': S0, 'I_P': IP0, 'I_Pa': IPa0})

    del p
    del a
    del q
    del N

    t, result = model.simulate(tmax = time , sampling_dt = sampling_dt)

    return t, result['C'], result['R'], result['Ra'],result['X'],result['Xa']
import csv
with open('mixeda=30.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["t", "run_id","C", "R", "Ra", "X", "Xa"])
    for i in range(20):
        t, C, R, Ra,X,Xa = mixed(N, parameter, time, sampling_dt, a, q,z, R0,)
        for j in range(len(t)):
            writer.writerow([t[j], i, C[j], R[j], Ra[j],X[j],Xa[j]])
