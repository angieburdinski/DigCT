from smallworld.draw import draw_network
from smallworld import get_smallworld_graph

from expon_smallworld import get_expon_random_model, get_expon_small_world

import matplotlib.pyplot as pl

# define network parameters
N = 100
labels = [ r'lattice', r'SW local clust. strong', 'SW local clust. weak', r'random']

N = 200
k0 = 5
k_over_2 = k0/2

Gs = [
        get_expon_small_world(N,k0,node_creation_order='desc'),
        get_expon_small_world(N,k0,node_creation_order='asc'),
        get_expon_small_world(N,k0,node_creation_order='random'),
        get_expon_random_model(N,k0),
     ]

focal_node = 0

fig, ax = pl.subplots(1,4,figsize=(11,3))


# scan beta values
for iG, G in enumerate(Gs):

    # generate small-world graphs and draw
    draw_network(G,k_over_2,focal_node=focal_node,ax=ax[iG],markersize=3)

    ax[iG].set_title(labels[iG],fontsize=11)

# show
pl.subplots_adjust(wspace=0.3)
pl.show()
