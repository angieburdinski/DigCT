import netwulf as nw
from expon_smallworld import get_expon_small_world

N = 20
k0 = 3
G = get_expon_small_world(N,k0,node_creation_order='asc')

nw.visualize(G)
