from smallworld.draw import draw_network
from smallworld import get_smallworld_graph

from expon_smallworld import get_expon_random_model, get_expon_small_world

import numpy as np
import matplotlib.pyplot as pl
import bfmplot as bp

# define network parameters
N = 100
labels = [ r'lattice', r'SW local clust. strong', 'SW local clust. weak', r'random']

N = 100_000
k0 = 20
k_over_2 = k0/2

Gs = [
        get_expon_small_world(N,k0,node_creation_order='desc'),
        get_expon_small_world(N,k0,node_creation_order='asc'),
        get_expon_small_world(N,k0,node_creation_order='random'),
        get_expon_random_model(N,k0),
     ]

focal_node = 0

fig, ax = pl.subplots(1,4,figsize=(11,3))

def get_distances(G):
    rs = []
    N = G.number_of_nodes()
    for u, v in G.edges():
        d = np.abs(u-v)
        d = min(d,N-d)
        rs.append(d)
    return rs

# scan beta values
for iG, G in enumerate(Gs):


    dists = np.array(get_distances(G))
    ax[iG].hist(dists,bins=20,density=True)
    ax[iG].set_yscale('log')
    ax[iG].set_xscale('log')
    ax[iG].set_xlabel('link distance (Hamming)')
    ax[iG].set_ylabel('pdf')
    ax[iG].set_title(labels[iG],fontsize=11)
    bp.strip_axis(ax[iG])

# show
#pl.subplots_adjust(wspace=0.3)
fig.tight_layout()
pl.show()
