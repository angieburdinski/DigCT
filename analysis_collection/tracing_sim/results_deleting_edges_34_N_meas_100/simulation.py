import numpy as np
import epipack as epk
import numpy as np
import networkx as nx

def make_equal_length(arr_list):
    maxlen = max([len(a) for a in arr_list])
    new_arr_list = []
    for a in arr_list:
        dL = maxlen - len(a)
        if dL > 0:
            newa = np.concatenate((a, np.ones(dL)*a[-1]))
        else:
            newa = a
        new_arr_list.append(newa)
    return new_arr_list

def simulation_code(kw):

    a = kw['a']
    q = kw['q']
    I0 = kw['I0_prob']
    k0 = kw['k0']


    kappa =  kw["rho"] * q/(1-q)

    N = kw['N']
    p = kw['k0'] / (N-1)
    G = nx.fast_gnp_random_graph(N, p)
    edges = [ (u,v,1.) for u, v in G.edges() ]
    k_norm = 2*len(edges)/N
    #tmaxs = [40,40,40,1e300]
    #Rscale = [1.0,0.4,1.0,0.4]
    #tmaxs = [1e300]
    #Rscale = [1.0]
    tmaxs = kw['phases'][kw['phase']]['tmaxs']
    Rscale = kw['phases'][kw['phase']]['Rscale']

    delete_edges_instead_of_scaling_R = kw['delete_edges_instead_of_scaling_R']

    _I0 = int(N*I0)
    _I0a = int(a*_I0)
    _I0 -= _I0a
    _S0 = N - _I0 - _I0a
    _S0a = int(_S0*a)
    _S0 -= _S0a

    node_statuses = None
    timebin_ts = []
    timebin_results = []
    last_t = 0

    #print(len(edges))

    if delete_edges_instead_of_scaling_R:
        ndx = np.random.permutation(len(edges))
        scrambled_edges = [ edges[i] for i in ndx ]

    for iphase, (this_tmax, this_Rscale) in enumerate(zip(tmaxs, Rscale)):

        if delete_edges_instead_of_scaling_R:
            these_edges = scrambled_edges[:int(this_Rscale*len(edges))]
            this_Rscale = 1
            #print(len(these_edges))
        else:
            these_edges = edges

        #model = epk.StochasticEpiModel([S,E,I,R,X,Sa,Ea,Ia,Ra,Xa,Ya,Za],N,edge_weight_tuples=these_edges)\

        model = epk.StochasticEpiModel(['S','E','I_P','I_S','I_A','R','T','X','Sa','Ea','I_Pa','I_Sa','I_Aa','Ra','Ta','Xa','Qa','C'],N, edge_weight_tuples=these_edges ,)
        model.set_conditional_link_transmission_processes({
            ("Ta", "->", "Xa") : [
                    ("Xa", "I_Pa", kw["y"], "Xa", "Ta" ),
                    ("Xa", "I_Sa", kw["y"], "Xa", "Ta" ),
                    ("Xa", "I_Aa", kw["y"], "Xa", "Ta" ),
                    ("Xa", "Ea", kw["y"], "Xa", "Ta" ),
                    ("Xa", "Sa", "->", "Xa", "Qa" ),
                    ("Xa", "I_Pa", (1-kw["y"]), "Xa", "C" ),
                    ("Xa", "I_Sa", (1-kw["y"]), "Xa", "C" ),
                    ("Xa", "I_Aa", (1-kw["y"]), "Xa", "C" ),
                    ("Xa", "Ea", (1-kw["y"]), "Xa", "C" )]
                    })
        model.set_node_transition_processes([
                    ('E',kw['alpha'],'I_P'),
                    ('I_P',(1-kw['x'])*kw['beta'],'I_S'),
                    ('I_P',kw['x']*kw['beta'],'I_A'),
                    ('I_A',kw['rho'],'R'),
                    ('I_S',kw['rho'],'R'),
                    ('I_S',kappa,'T'),
                    ('T',kw['chi'],'X'),
                    ('Qa',kw['omega'],'Sa'),
                    ('Ea',kw['alpha'],'I_Pa'),
                    ('I_Pa',(1-kw['x'])*kw['beta'],'I_Sa'),
                    ('I_Pa',kw['x']*kw['beta'],'I_Aa'),
                    ('I_Aa',kw['rho'],'Ra'),
                    ('I_Sa',kw['rho'],'Ra'),
                    ('I_Sa',kappa,'Ta'),
                    ('Ta',kw["z"]*kw['chi'],'Xa'),
                    ('Ta',(1-kw["z"])*kw['chi'],'X')])
        model.set_link_transmission_processes([

                    ('I_Pa','S',kw["R0"]/k_norm*kw['beta']/2,'I_Pa','E'),
                    ('I_Aa','S',kw["R0"]/k_norm*kw['rho']/2,'I_Aa','E'),
                    ('I_Sa','S',kw["R0"]/k_norm*kw['rho']/2,'I_Sa','E'),

                    ('I_P','Sa',kw["R0"]/k_norm*kw['beta']/2,'I_P','Ea'),
                    ('I_A','Sa',kw["R0"]/k_norm*kw['rho']/2,'I_A','Ea'),
                    ('I_S','Sa',kw["R0"]/k_norm*kw['rho']/2,'I_S','Ea'),

                    ('I_Pa','Sa',kw["R0"]/k_norm*kw['beta']/2,'I_Pa','Ea'),
                    ('I_Aa','Sa',kw["R0"]/k_norm*kw['rho']/2,'I_Aa','Ea'),
                    ('I_Sa','Sa',kw["R0"]/k_norm*kw['rho']/2,'I_Sa','Ea'),

                    ('I_P','S',kw["R0"]/k_norm*kw['beta']/2,'I_P','E'),
                    ('I_A','S',kw["R0"]/k_norm*kw['rho']/2,'I_A','E'),
                    ('I_S','S',kw["R0"]/k_norm*kw['rho']/2,'I_S','E')])

        if node_statuses is None:
            model.set_random_initial_conditions({
                           "Sa": _S0a,
                           "I_Pa": _I0a,
                           "S": _S0,
                           "I_P": _I0,
                       })
        else:
            model.set_node_statuses(node_statuses)

        this_t, this_result = model.simulate(this_tmax+last_t,sampling_dt=1,t0=last_t)

        if iphase < len(tmaxs)-1:
            this_t = this_t[:-1]
            this_result = { C: arr[:-1] for C, arr in this_result.items() }
        last_t += this_tmax

        node_statuses = model.node_status

        timebin_ts.append(this_t)
        timebin_results.append(this_result)

    if len(tmaxs) > 1:
        t = np.concatenate(timebin_ts)
        this_result = { C: np.concatenate([res[C] for res in timebin_results]) for C in model.compartments }
    else:
        t = timebin_ts[0]
        this_result = timebin_results[0]

    return this_result

if __name__ == "__main__":

    import matplotlib.pyplot as pl
    import qsuite_config as cf
    from pprint import pprint

    kw = {}
    for p in cf.external_parameters + cf.internal_parameters:
        if p[0] is not None:
            kw[p[0]] = p[1][0]

    for p in cf.standard_parameters:
        if p[0] is not None:
            kw[p[0]] = p[1]

    kw['phase'] = 'periodic lockdown'
    kw['N'] = 200000
    kw['chi'] = 1/2.5
    kw['rho'] =  1/7
    kw['alpha'] = 1/3
    kw['beta'] = 1/2
    kw['k0'] =  20
    kw['x'] = 0.17
    kw['I0_prob'] = 0.001
    kw['omega'] = 1/10
    kw['z'] = 0.64
    kw['R0'] = 2.5
    kw['y'] = 0.1

    print("using config:")
    pprint(kw)
    print()

    result = simulation_code(kw)

    from epipack.plottools import plot

    t = np.arange(len(result['S']))
    #plot(t, result["I_P"])
    pl.plot(t, result["I_P"]+result["I_Pa"])

    pl.show()
