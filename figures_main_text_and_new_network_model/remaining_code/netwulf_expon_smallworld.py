import netwulf as nw
from expon_smallworld import get_expon_small_world

N = 1000
k0 = 10
G = get_expon_small_world(N,k0,node_creation_order='desc')

nw.visualize(G)

N = 1000
k0 = 10
G = get_expon_small_world(N,k0,node_creation_order='random')

nw.visualize(G)

N = 1000
k0 = 10
G = get_expon_small_world(N,k0,node_creation_order='asc')

nw.visualize(G)
