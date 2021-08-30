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

    S, E, I, R, X = list("SEIRX")
    Sa, Ea, Ia, Ra, Xa = [letter+"a" for letter in "SEIRX"]
    Za = "Za"
    Ya = "Ya"

    a = kw['a']
    q = kw['q']

    k0 = kw['k0']
    I0 = kw['I0_prob']
    alpha = kw['alpha']
    R0 = kw['R0']
    rho = kw['rho']

    kappa =  rho * q/(1-q)


    N = kw['N']
    p = kw['k0'] / (N-1)
    G = nx.fast_gnp_random_graph(N, p)
    edges = [ (u,v,1.) for u, v in G.edges() ]

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

    if delete_edges_instead_of_scaling_R:
        ndx = np.random.permutation(len(edges))
        scrambled_edges = [ edges[i] for i in ndx ]

    for iphase, (this_tmax, this_Rscale) in enumerate(zip(tmaxs, Rscale)):

        if delete_edges_instead_of_scaling_R:
            these_edges = scrambled_edges[:int(this_Rscale*len(edges))]
            this_Rscale = 1
        else:
            these_edges = edges

        model = epk.StochasticEpiModel([S,E,I,R,X,Sa,Ea,Ia,Ra,Xa,Ya,Za],N,edge_weight_tuples=these_edges)\
                   .set_node_transition_processes([
                           ("Ea", alpha, "Ia"),
                           ("Ia", rho,"Ra"),
                           ("Ia", kappa, "Xa"),
                           ("E", alpha, "I"),
                           ("I", rho,"R"),
                           ("I", kappa, "X"),
                       ])\
                   .set_link_transmission_processes([
                           ("Ia", "Sa", R0*rho/k0*this_Rscale, "Ia", "Ea"),
                           ("Ia", "S", R0*rho/k0*this_Rscale, "Ia", "E"),
                           ("I", "Sa", R0*rho/k0*this_Rscale, "I", "Ea"),
                           ("I", "S", R0*rho/k0*this_Rscale, "I", "E"),
                       ])\
                   .set_conditional_link_transmission_processes({
                        ( "Ia", "->", "Xa" ) : [
                                ("Xa", "Ia", "->", "Xa", "Ya"),
                                ("Xa", "Ea", "->", "Xa", "Za"),
                            ]
                       })
        if node_statuses is None:
            model.set_random_initial_conditions({
                           "Sa": _S0a,
                           "Ia": _I0a,
                           "S": _S0,
                           "I": _I0,
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
    kw['N'] = 20000

    print("using config:")
    pprint(kw)
    print()

    result = simulation_code(kw)

    from epipack.plottools import plot

    t = np.arange(len(result['S']))
    plot(t, result)

    pl.show()
