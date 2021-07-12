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

def get_expon_random_model(N,k0):

    degree_seq = [ int(k) for k in expon.rvs(scale=k0,size=N)]
    stubs = list(degree_seq)
    if sum(stubs) % 2 == 1:
        stubs[np.random.randint(0,N-1)] += 1

    G = nx.configuration_model(stubs)
    G = nx.Graph(G)

    return G

def get_degree_stats(G):
    k = [ G.degree(n) for n in G.nodes() ]
    return np.mean(k), np.std(k)

if __name__ == "__main__":

    import smallworld as sw

    N = 20_000
    k0 = 20

    Gs = [
            get_expon_small_world(N,k0,node_creation_order='desc'),
            get_expon_small_world(N,k0,node_creation_order='asc'),
            get_expon_small_world(N,k0,node_creation_order='random'),
            get_expon_random_model(N,k0),
         ]

    for G in Gs:
        print(get_degree_stats(G), k0, k0)



    ts = []
    Cs = []
    As = []

    for G in Gs:

        Gcomp = sw.tools.get_largest_component(G)
        t = 1/sw.tools.get_random_walk_eigenvalue_gap(nx.to_scipy_sparse_matrix(Gcomp,dtype=float))
        C = np.mean(list(nx.clustering(G).values()))
        A = nx.degree_assortativity_coefficient(G)
        ts.append(t)
        Cs.append(C)
        As.append(A)
        print("relaxation time =",t)
        print("clustering =",C)
        print("clustering large comp =",np.mean(list(nx.clustering(Gcomp).values())))
        print("degree assortativity =",A)
        print()

    import matplotlib.pyplot as pl

    fig, ax = pl.subplots()
    ax.plot(ts,marker='o')
    ax2 = ax.twinx()
    ax2.plot(Cs,marker='s')
    ax2.set_ylabel('C')
    ax.set_ylabel('relaxation time')


    pl.show()

