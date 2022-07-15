import epipack
import numpy as np
from epipack.stochastic_epi_models import StochasticEpiModel
from math import exp
from numpy import random
import networkx as nx
import matplotlib.ticker as mtick
from smallworld import get_smallworld_graph
from scipy.stats import expon
import matplotlib.pyplot as plt
def ER_network(N,p):
    G = get_smallworld_graph(N, int(p["k0"]/2), beta = 1)
    edge_weight_tuples = [ (e[0], e[1], 1.0) for e in G.edges() ]
    k_norm = 2*len(edge_weight_tuples) / N
    del G
    return edge_weight_tuples, k_norm
def old_model(p,a, edge_weight_tuples, k_norm):
    kappa = (p['q']*p['rho'])/(1-p['q'])
    IPa0 = int(random.binomial(p['I_0'], a, 1))
    IP0 = int(p['I_0'] - IPa0)
    Sa0 = int(random.binomial(p['N']-p['I_0'], a, 1))
    S0 = int(p['N'] - p['I_0'] - Sa0)

    model = epipack.StochasticEpiModel(['S','E','I_P','I_S','I_A','R','T','X','Sa','Ea','I_Pa','I_Sa','I_Aa','Ra','Ta','Xa','Qa','C'],p['N'], edge_weight_tuples ,directed=False)
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
                ('I_A',p['rho'],'R'),
                ('I_S',p['rho'],'R'),
                ('I_S',kappa,'T'),
                ('T',p['chi'],'X'),
                ('Qa',p['omega'],'Sa'),
                ('Ea',p['alpha'],'I_Pa'),
                ('I_Pa',(1-p['x'])*p['beta'],'I_Sa'),
                ('I_Pa',p['x']*p['beta'],'I_Aa'),
                ('I_Aa',p['rho'],'Ra'),
                ('I_Sa',p['rho'],'Ra'),
                ('I_Sa',kappa,'Ta'),
                ('Ta',p["z"]*p['chi'],'Xa'),
                ('Ta',(1-p["z"])*p['chi'],'X')])
    model.set_link_transmission_processes([

                ('I_Pa','S',p["R0"]/k_norm*p['beta']/2,'I_Pa','E'),
                ('I_Aa','S',p["R0"]/k_norm*p['rho']/2,'I_Aa','E'),
                ('I_Sa','S',p["R0"]/k_norm*p['rho']/2,'I_Sa','E'),

                ('I_P','Sa',p["R0"]/k_norm*p['beta']/2,'I_P','Ea'),
                ('I_A','Sa',p["R0"]/k_norm*p['rho']/2,'I_A','Ea'),
                ('I_S','Sa',p["R0"]/k_norm*p['rho']/2,'I_S','Ea'),

                ('I_Pa','Sa',p["R0"]/k_norm*p['beta']/2,'I_Pa','Ea'),
                ('I_Aa','Sa',p["R0"]/k_norm*p['rho']/2,'I_Aa','Ea'),
                ('I_Sa','Sa',p["R0"]/k_norm*p['rho']/2,'I_Sa','Ea'),

                ('I_P','S',p["R0"]/k_norm*p['beta']/2,'I_P','E'),
                ('I_A','S',p["R0"]/k_norm*p['rho']/2,'I_A','E'),
                ('I_S','S',p["R0"]/k_norm*p['rho']/2,'I_S','E')])
    model.set_network(p['N'], edge_weight_tuples)
    model.set_random_initial_conditions({ 'Sa': Sa0, 'S': S0, 'I_P': IP0, 'I_Pa': IPa0})
    t, result = model.simulate(tmax = p["time"] , sampling_dt = p["sampling_dt"])

    return  max(result['R'])+max(result['Ra'])+ max(result['X'])+ max(result['Xa']) + max(result['C'])
def new_model(p,a, edge_weight_tuples, k_norm):

    kappa = p["c"] * p["zeta"]
    rho_1 = (1-p["c"]) * p["zeta"]
    rho_2 = 1/ ( 1/p["rho"] - 1/rho_1)

    rho_1_base =  p["zeta"]
    rho_2_base = 1/ ( 1/p["rho"] - 1/rho_1_base)
    R0_I1 = p["R0"]/2 * (1/rho_1_base) / (1/p["rho"])
    R0_I2 = p["R0"]/2 * (1/rho_2_base) / (1/p["rho"])
    IPa0 = int(random.binomial(p['I_0'], a, 1))
    IP0 = int(p['I_0'] - IPa0)
    Sa0 = int(random.binomial(p['N']-p['I_0'], a, 1))
    S0 = int(p['N'] - p['I_0'] - Sa0)

    model = epipack.StochasticEpiModel(['S','E','I_P','I_S1','I_S2','I_A','R','T','X','Sa','Ea','I_Pa','I_Sa1','I_Sa2','I_Aa','Ra','Ta','Xa','Qa','C'],p['N'], edge_weight_tuples ,directed=False)
    model.set_conditional_link_transmission_processes({
    ("Ta", "->", "Xa") : [
            ("Xa", "I_Pa", p["y"], "Xa", "Ta" ),
            ("Xa", "I_Sa1", p["y"], "Xa", "Ta" ),
            ("Xa", "I_Sa2", p["y"], "Xa", "Ta" ),
            ("Xa", "I_Aa", p["y"], "Xa", "Ta" ),
            ("Xa", "Ea", p["y"], "Xa", "Ta" ),
            ("Xa", "Sa", "->", "Xa", "Qa" ),
            ("Xa", "I_Pa", (1-p["y"]), "Xa", "C" ),
            ("Xa", "I_Sa1", (1-p["y"]), "Xa", "C" ),
            ("Xa", "I_Sa2", (1-p["y"]), "Xa", "C" ),
            ("Xa", "I_Aa", (1-p["y"]), "Xa", "C" ),
            ("Xa", "Ea", (1-p["y"]), "Xa", "C" )]
            })
    model.set_node_transition_processes([
                ('E',p['alpha'],'I_P'),
                ('I_P',(1-p['x'])*p['beta'],'I_S1'),
                ('I_P',p['x']*p['beta'],'I_A'),
                ('I_A',p['rho'],'R'),

                ('I_S1',rho_1,'I_S2'),
                ('I_S2',rho_2,'R'),
                ('I_S1',kappa,'T'),

                ('T',p['chi'],'X'),
                ('Qa',p['omega'],'Sa'),
                ('Ea',p['alpha'],'I_Pa'),
                ('I_Pa',(1-p['x'])*p['beta'],'I_Sa1'),
                ('I_Pa',p['x']*p['beta'],'I_Aa'),
                ('I_Aa',p['rho'],'Ra'),

                ('I_Sa1',rho_1,'I_Sa2'),
                ('I_Sa2',rho_2,'Ra'),
                ('I_Sa1',kappa,'Ta'),

                ('Ta',p["z"]*p['chi'],'Xa'),
                ('Ta',(1-p["z"])*p['chi'],'X')])
    model.set_link_transmission_processes([

                ('I_Pa','S',p["R0"]/2/k_norm*p['beta'],'I_Pa','E'),
                ('I_Aa','S',p["R0"]/2/k_norm*p['rho'],'I_Aa','E'),
                ('I_Sa1','S',R0_I1/k_norm*rho_1_base,'I_Sa1','E'),
                ('I_Sa2','S',R0_I2/k_norm*rho_2_base,'I_Sa2','E'),

                ('I_P','Sa',p["R0"]/2/k_norm*p['beta'],'I_P','Ea'),
                ('I_A','Sa',p["R0"]/2/k_norm*p['rho'],'I_A','Ea'),
                ('I_S1','Sa',R0_I1/k_norm*rho_1_base,'I_S1','Ea'),
                ('I_S2','Sa',R0_I2/k_norm*rho_2_base,'I_S2','Ea'),

                ('I_Pa','Sa',p["R0"]/2/k_norm*p['beta'],'I_Pa','Ea'),
                ('I_Aa','Sa',p["R0"]/2/k_norm*p['rho'],'I_Aa','Ea'),
                ('I_Sa1','Sa',R0_I1/k_norm*rho_1_base,'I_Sa1','Ea'),
                ('I_Sa2','Sa',R0_I2/k_norm*rho_2_base,'I_Sa2','Ea'),

                ('I_P','S',p["R0"]/2/k_norm*p['beta'],'I_P','E'),
                ('I_A','S',p["R0"]/2/k_norm*p['rho'],'I_A','E'),
                ('I_S1','S',R0_I1/k_norm*rho_1_base,'I_S1','E'),
                ('I_S2','S',R0_I2/k_norm*rho_2_base,'I_S2','E')])

    model.set_network(p['N'], edge_weight_tuples)
    model.set_random_initial_conditions({ 'Sa': Sa0, 'S': S0, 'I_P': IP0, 'I_Pa': IPa0})
    t, result = model.simulate(tmax = p["time"] , sampling_dt = p["sampling_dt"])
    return max(result['R'])+max(result['Ra'])+ max(result['X'])+ max(result['Xa']) + max(result['C'])

N = 10_000
a_s = np.linspace(0,1,5)
xticks = [str(int(a_s[i]*100)) +'%' for i in range(1,len(a_s))]
it = 100
detection = [0.1,0.3,0.5]
def outbr_red(x0_vals, x1_vals):

    n = len(x0_vals)
    m = len(x1_vals)

    x0 = np.mean(x0_vals)
    #err_x0 = np.std(x0_vals)/np.sqrt(n)
    err_x0 = np.std(x0_vals)
    x1 = np.mean(x1_vals)
    #err_x1 = np.std(x1_vals)/np.sqrt(m)
    err_x1 = np.std(x1_vals)

    red = x1/x0 - 1

    delta_0 = err_x0 * x1/x0**2
    delta_1 = err_x1 * 1/x0

    err_red = np.sqrt(delta_0**2 + delta_1**2)

    return red, err_red
fig,ax = plt.subplots(1,2,sharey = True)
for i_d,d in enumerate(detection[1:]):
    O = np.zeros((2,len(a_s),it))
    p = {
            'chi':1/2.5,
            'rho' : 1/20,
            'alpha' : 1/3,
            'beta' : 1/2,
            'k0' : 20,
            'x':0.17,
            'I_0' : N*0.01,
            'omega':1/10,
            "y" : 0.5,
            "z": 0.64,
            "N": N,
            "quarantiningS": True,
            "q": d,
            "R0": 2.5,
            "sampling_dt": 1,
            "time" : 10e6,
            "c": d,
            "zeta": 1
            }

    for i_a, a in enumerate(a_s):
        for i in range(it):
            edge_weight_tuples, k_norm = ER_network(p['N'], p)
            O[0,i_a,i]  = old_model(p,a,edge_weight_tuples, k_norm)
            O[1,i_a,i]  = new_model(p,a,edge_weight_tuples, k_norm)

        if i_a > 0 :
            red, err_red = outbr_red(O[0,0],O[0,i_a])
            ax[i_d].bar(i_a ,red*100, yerr = err_red*100, width = 0.2,color = 'grey')
            red, err_red = outbr_red(O[1,0],O[1,i_a])
            ax[i_d].bar(i_a + 0.2 ,red*100 , yerr = err_red*100, width = 0.2,color = 'navy')
for i in range(2):
    ax[i].set_xticks([1,2,3,4])
    ax[i].set_xticklabels(xticks)
    ax[i].set_ylabel('outbreaksize reduction')
    ax[i].yaxis.set_major_formatter(mtick.PercentFormatter())
    ax[i].set_xlabel('app participation')
plt.legend()
plt.show()
