from extendedmodel import stoch_mixed_tracing
from tools import configuration_network
from plots import plot
import networkx as nx
import numpy as np
from tools import analysis
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
color =  dict(mcolors.TABLEAU_COLORS, **mcolors.CSS4_COLORS)
colors = [i for i in color.keys()]

N = 10000
k0 = 19
G = configuration_network(N,k0).build()
t = 10000
model = stoch_mixed_tracing(G,quarantine_S_contacts = True)
parameter = {
        'R0': 2.5,
        'q': 0.5,
        'app_participation': 0.33,
        'chi':1/2.5,
        'recovery_rate' : 1/6,
        'alpha' : 1/2.5,
        'beta' : 1/2.5,
        'number_of_contacts' : 6.3,
        'x':0.17,
        'y':0.1,
        'z':0.64,
        'I_0' : 100,
        'omega':1/10
        }
from matplotlib import rc
plt.rcParams.update({'font.size': 13})
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
q = [0,0.2,0.4,0.6,0.8]

a = np.linspace(0,0.8,10)
t, result = analysis(model,parameter).stoch_two_range_result('app_participation',a,'q',q,t)
print(result)
fig, axs = plt.subplots(1,3,figsize = (18,3))
for i in a:
    for j in q:
        axs[0].plot(i, (result[i][j]['R'].max(axis = 0)+result[i][j]['Ra'].max(axis = 0)+result[i][j]['Xa'].max(axis = 0)+result[i][j]['X'].max(axis = 0))/N,'.', color = colors[q.index(j)])
        axs[1].plot(i, (result[i][j]['I_S'].max(axis = 0)+result[i][j]['I_Sa'].max(axis = 0))/N,'.', color = colors[q.index(j)])
        axs[2].plot(i, (result[i][j]['Qa'].max(axis = 0))/N,'.', color = colors[q.index(j)])
axs[0].set_xlabel('app participation')
axs[1].set_xlabel('app participation')
axs[2].set_xlabel('app participation')
axs[0].set_ylabel(r'$R (t\rightarrow \infty)+X (t\rightarrow \infty)$')
axs[1].set_ylabel(r'$I_{S_{max}}$')
axs[2].set_ylabel(r'$Q_{max}$')
lines = [Line2D([0], [0], color=colors[x], linewidth=3, linestyle='dotted') for x in range(len(q))]
labels = [('q = ' + str(j)) for j in q]
fig.legend(lines, labels)
plt.show()
