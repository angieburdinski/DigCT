import epipack as ep
import numpy as np
from epipack.deterministic_epi_models import DeterministicEpiModel
from epipack.stochastic_epi_models import StochasticEpiModel
from numpy import random

def simulation_code(kwargs):

    def SEIRTX(N,a, q, rho,chi,eta,alpha):
        edges = np.loadtxt('facebook_combined.txt')
        kappa = (rho*q)/(1-q)
        I0 = 1
        Sa0 = int(random.binomial(N-I0, a, 1))
        S0 = N - I0 - Sa0
        weighted_edge_tuples =[]
        for i in range(88234):
            weighted_edge_tuples.append((int(edges[i,0]),int(edges[i,1]),1.0))
        k_norm = 2*len(weighted_edge_tuples) / N

        model = StochasticEpiModel(["S","E","I","R","T","X","Sa","Ea","Ia","Ra","Ta","Xa"],N,edge_weight_tuples=weighted_edge_tuples)
        model.set_node_transition_processes([
                ("I", rho, "R" ),
                ("Ia", rho, "Ra" ),
                ("I", kappa, "T" ),
                ("Ia", kappa, "Ta" ),
                ("T", chi, "X" ),
                ("Ta", chi, "Xa" ),
                ("E", alpha, "I" ),
                ("Ea", alpha, "Ia" ),

            ])
        model.set_link_transmission_processes([
                ("I", "S", eta/k_norm, "I", "E" ),
                ("Ia", "S", eta/k_norm, "Ia", "E" ),
                ("I", "Sa", eta/k_norm, "I", "Ea" ),
                ("Ia", "Sa", eta/k_norm, "Ia", "Ea" ),
             ])
        model.set_conditional_link_transmission_processes({
    ("Ta", "->", "Xa") : [
                ("Xa", "Ia", "->", "Xa", "Ta" ),
                ("Xa", "Ea", "->", "Xa", "Ta" )]
})
        model.set_network(N,weighted_edge_tuples)
        model.set_random_initial_conditions({"S":S0,"I":I0, 'Sa':Sa0})

        t, result = model.simulate(10000)

        return t, result

    t, result = SEIRTX(**kwargs)

    results = (max(result['I'])+max(result['Ia']),max(result['R'])+max(result['Ra'])+max(result['X'])+max(result['Xa']))
    return results
