import pickle
import numpy as np
import matplotlib.pyplot as pl
import matplotlib as mpl
from test_layout_alg import draw_network
from expon_smallworld import get_expon_small_world, get_expon_random_model
from cycler import cycler

import bfmplot as bp

with open('data_from_analysis.p','rb') as f:
    data = pickle.load(f)
#with open('test_data.p','rb') as f:
#    data = pickle.load(f)



fig = pl.figure(figsize=(8,8))
axN = [ pl.subplot(4,4,1+i) for i in range(4) ]
axCt = pl.subplot(412)
#axAN = pl.subplot(513)
axK = [ pl.subplot(4,4,2*4+1+i) for i in range(4) ]
axD = [ pl.subplot(4,4,3*4+1+i) for i in range(4) ]

def get_ccdf(d):
    d = np.sort(d)
    return d, 1 - np.arange(len(d),dtype=float)/(len(d)-1)

N = 5_000
k0 = 10

bp.colors = ['#333333','#888888']
mpl.rcParams['axes.prop_cycle'] = cycler(color=bp.colors)

show_networks = False
show_networks = True

if show_networks:
    Gs = [
            get_expon_small_world(N,k0,node_creation_order='desc'),
            get_expon_small_world(N,k0,node_creation_order='asc'),
            get_expon_small_world(N,k0,node_creation_order='random'),
            get_expon_random_model(N,k0),
         ]

labels = ['SW lattice\n(desc)', 'SW assortative\n(asc)','SW random\n(random)','random\n(conf. model)']
for i in range(4):
    ax = axN[i]
    if show_networks:
        G = Gs[i]
        draw_network(G,ax,max_size=40,expon=1.2,lw=0.2)
    ax.text(0.99,0.01,'deg. assort. = {0:4.3f}'.format(data['degree_assortativity'][i]),
            ha='right',
            va='top',
            transform=ax.transAxes)

    ax.set_title(labels[i],fontsize='medium')

print(data.keys())
axC = axCt.twinx()
axC.plot(data['average_clustering_coefficients'],'-s',color=bp.colors[0])
axCt.plot(np.array(data['relaxation_times'])/(len(data['distances'][0])),'-o',c=bp.colors[1])
axCt.set_ylabel('relaxation time t/N')
axC.set_ylabel('average clustering coeff. C')
axC.set_ylim([-0.02,0.62])
axCt.set_ylim([-0.02,0.62])
axCt.spines['top'].set_visible(False)
axCt.spines['bottom'].set_visible(False)
axC.spines['top'].set_visible(False)
axC.spines['bottom'].set_visible(False)
axCt.set_xticks([])

axCt.text(0.3,0.3,'$t/N$',transform=axCt.transAxes)
axCt.text(0.75,0.5,'$C$',transform=axCt.transAxes)

axCt.set_xlim([-0.3,3.3])

maxk = max([max(d) for d in data['degrees']])
maxd = max([max(d) for d in data['distances']])
N = len(data['degrees'][0])

es = [1.3,1,1,None]
thr = [100,10,10]
fac = [5,30,30]
lblpos = [0.6,0.7,0.7]
for i in range(4):
    a0 = axK[i]
    a1 = axD[i]
    e = es[i]
    bp.strip_axis(a0)
    bp.strip_axis(a1)
    x, P = get_ccdf(data['degrees'][i])
    a0.step(x,P,c=bp.colors[0])
    x, P = get_ccdf(data['distances'][i])
    #x = np.array(x,dtype=float)
    #x /= (N/2)
    a1.step(x,P,c=bp.colors[0])
    if e is not None:
        x_ = x[x>thr[i]]
        a1.plot(x_,fac[i]*np.array(x_,dtype=float)**(-e),lw=2,ls='--',c=bp.colors[1])
    a0.set_yscale('log')
    a1.set_xscale('log')
    a1.set_yscale('log')

    a0.set_xlim([0,maxk])
    a1.set_xlim([1,N])
    a0.set_xticks([0,100,200,300])
    a1.set_xticks([1,10**2,10**4,])
    #bp.set_n_ticks(a0,nx=4,ny=None)
    #bp.set_n_ticks(a1,nx=4,ny=None)

    a0.set_ylabel('ccdf')
    a1.set_ylabel('ccdf')
    a0.set_xlabel('degree')
    a1.set_xlabel('edge distance d(i,j)')

    if e is not None:
        a1.text(lblpos[i],lblpos[i],'$d^{-%4.1f}$' % (e),transform=a1.transAxes,color='#555555')

#axA = axAN.twinx()
#axA.plot(data['degree_assortativity'],'-d',c=bp.colors[2])
#axAN.plot(data['sizes_largest_component'],'-x',c=bp.colors[3])
fig.tight_layout()
fig.savefig('sw_expon_model.png',dpi=300)

pl.show()
