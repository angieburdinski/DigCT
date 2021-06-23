from scipy.stats import expon
import numpy as np
import networkx as nx

from expon_smallworld import get_expon_small_world, get_expon_random_model
import smallworld as sw

def dist(i,j,N):
    return np.min([np.abs(i-j),N-np.abs(i-j)])

def get_degree_stats(G):
    k = [ G.degree(n) for n in G.nodes() ]
    return k, np.mean(k), np.std(k)

if __name__ == "__main__":


    N = 100_000
    k0 = 20

    Gs = [
            get_expon_small_world(N,k0,node_creation_order='desc'),
            get_expon_small_world(N,k0,node_creation_order='asc'),
            get_expon_small_world(N,k0,node_creation_order='random'),
            get_expon_random_model(N,k0),
         ]

    for G in Gs:
        print(get_degree_stats(G)[1:], k0, k0)



    ts = []
    Cs = []
    As = []
    Ns = []
    ks = []
    distances = []
    graphs = Gs
    order = ['lattice','expon_assortative','expon_random','random']

    for G in Gs:

        Gcomp = sw.tools.get_largest_component(G)
        Ncomp = Gcomp.number_of_nodes()

        t = 1/sw.tools.get_random_walk_eigenvalue_gap(nx.to_scipy_sparse_matrix(Gcomp,dtype=float))
        C = np.mean(list(nx.clustering(G).values()))
        A = nx.degree_assortativity_coefficient(G)
        d = [ dist(i,j,N) for i, j in G.edges() ]
        k,_,__ = get_degree_stats(G)

        ts.append(t)
        Cs.append(C)
        As.append(A)
        Ns.append(Ncomp)
        distances.append(d)
        ks.append(k)
        print("relaxation time =",t)
        print("clustering =",C)
        print("clustering large comp =",np.mean(list(nx.clustering(Gcomp).values())))
        print("degree assortativity =",A)
        print()

    wrapped = {
                'relaxation_times' : ts,
                'average_clustering_coefficients': Cs,
                'degree_assortativity': As,
                'sizes_largest_component': Ns,
                #'graphs' : graphs,
                'labels' : order,
                'distances' : distances,
                'degrees' : ks,
              }

    import pickle as pickle

    with open('data_from_analysis.p','wb') as f:
        pickle.dump(wrapped,f)

    import matplotlib.pyplot as pl

    fig, ax = pl.subplots()
    ax.plot(ts,marker='o')
    ax2 = ax.twinx()
    ax2.plot(Cs,marker='s')
    ax2.set_ylabel('C')
    ax.set_ylabel('relaxation time')


    pl.show()

