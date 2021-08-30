import numpy as np
from expon_smallworld import get_expon_small_world
import networkx as nx
#import netwulf as nw

def d(i,j,N):
    return min(np.abs(i-j), N-np.abs(i-j))

def get_layout(G,nbounce=20,Rbounce=0.1,R=1,scale_exponent=1):

    node_distances = [ [] for n in sorted(G.nodes()) ]
    N = G.number_of_nodes()

    for i, j in G.edges():
        node_distances[i].append(d(i,j,N))
        node_distances[j].append(d(i,j,N))

    node_distance_means = [ np.mean(dists) for dists in node_distances ]
    node_distance_maxs = [ np.max(dists+[0]) for dists in node_distances ]
    distance_measure = node_distance_maxs

    dphi = 2*np.pi/N
    maxmean = max(distance_measure)
    maxmean = max(3*N/4, maxmean)
    #maxmean = np.sum(1/np.arange(1,N+1)) / np.sum(1/np.arange(1,N+1)**2)
    #print(maxmean)

    pos = {}

    for n in G.nodes():
        phi = n*dphi
        rbase = R+Rbounce*np.sin(nbounce*phi)
        scl = (1-(distance_measure[n]/maxmean)**scale_exponent)
        r = rbase*(1-distance_measure[n]/maxmean)
        x = r*np.cos(phi)
        y = r*np.sin(phi)
        pos[n] = [x,y]

    return pos

def get_node_sizes(G,expon=0.5,max_size=5):
    node_degrees = np.array([ G.degree(n) for n in sorted(G.nodes()) ],dtype=float)
    max_degree = node_degrees.max()
    node_degrees[node_degrees==0] = 0.5
    node_degrees /= max_degree
    sizes = node_degrees**expon * max_size
    return sizes


def draw_network(G,ax,max_size=300,expon=2,lw=0.4):

    ax.set_aspect('equal', adjustable='box')

    pos = get_layout(G)
    sizes = get_node_sizes(G,max_size=max_size,expon=expon)
    print(sizes)
    nx.draw_networkx_edges(
            G,
            pos=pos,
            ax=ax,
            alpha=0.3,
            edge_color='#999999',
            )
    nx.draw_networkx_nodes(
            G,
            pos=pos,
            ax=ax,
            node_color='#333333',
            edgecolors='#ffffff',
            linewidths=lw,
            node_size = sizes,
            )
    ax.set_axis_off()


if __name__ == "__main__":
    import matplotlib.pyplot as pl

    fig, ax = pl.subplots(figsize=(6,6))
    N = 10000
    k0 = 10
    draw_network(get_expon_small_world(N,k0,node_creation_order='random'),ax)
    pl.show()
