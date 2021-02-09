from extendedmodel import (first_generation_tracing,next_generation_tracing,mixed_tracing,stoch_mixed_tracing)
from tools import (analysis,configuration_network)
from plots import plot
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
color =  dict(mcolors.TABLEAU_COLORS, **mcolors.CSS4_COLORS)
colors = [i for i in color.keys()]

def mixed_tracing_test():
    N = 10_000
    t = np.linspace(0,500,500)
    model = mixed_tracing(N,quarantine_S_contacts = True)
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
            'I_0' : 10,
            'omega':1/10
            }
    q = [0,0.2,0.4,0.6,0.8]
    a = np.linspace(0,0.8,50)

    fig, axs = plt.subplots(2,4,figsize = (40,8))
    result = analysis(model,parameter).two_range_result('app_participation',a,'q',q,t)
    for i in a:
        for j in q:
            axs[0,0].plot(i, (result[i][j]['S'].max(axis = 0))/N,'.', color = colors[q.index(j)])
            axs[0,1].plot(i, (result[i][j]['E'].max(axis = 0))/N,'.', color = colors[q.index(j)])
            axs[0,2].plot(i, (result[i][j]['I_P'].max(axis = 0))/N,'.', color = colors[q.index(j)])
            axs[0,3].plot(i, (result[i][j]['I_S'].max(axis = 0))/N,'.', color = colors[q.index(j)])
            axs[1,0].plot(i, (result[i][j]['I_A'].max(axis = 0))/N,'.', color = colors[q.index(j)])
            axs[1,1].plot(i, (result[i][j]['T'].max(axis = 0))/N,'.', color = colors[q.index(j)])
            axs[1,2].plot(i, (result[i][j]['R'].max(axis = 0)+result[i][j]['X'].max(axis = 0))/N,'.', color = colors[q.index(j)])
            axs[1,3].plot(i, (result[i][j]['Q'].max(axis = 0))/N,'.', color = colors[q.index(j)])
    for i in range(4):
        axs[1,i].set_xlabel(r'$a$')

    axs[0,0].set_ylabel(r'$S_{max}$')
    axs[0,1].set_ylabel(r'$E_{max}$')
    axs[0,2].set_ylabel(r'$I_{P_{max}}$')
    axs[0,3].set_ylabel(r'$I_{S_{max}}$')
    axs[1,0].set_ylabel(r'$I_{A_{max}}$')
    axs[1,1].set_ylabel(r'$T_{max}$')
    axs[1,2].set_ylabel(r'$R (t\rightarrow \infty)+X (t\rightarrow \infty)$')
    axs[1,3].set_ylabel(r'$Q_{max}$')

    lines = [Line2D([0], [0], color=colors[x], linewidth=3, linestyle='dotted') for x in range(len(q))]
    labels = [('q = ' + str(j)) for j in q]
    fig.legend(lines, labels)
    plt.show()
def stoch_mixed_tracing_test(r):
    fig, axs = plt.subplots(2,4,figsize = (40,8))
    N = 10000
    k0 = 19
    time = 1000
    parameter = {
            'R0': 2.5,
            'q': 0.5,
            'app_participation': 0.33,
            'chi':1/2.5,
            'recovery_rate' : 1/6,
            'alpha' : 1/2.5,
            'beta' : 1/2.5,
            #'number_of_contacts' : 6.3,
            'x':0.17,
            'y':0.1,
            'z':0.64,
            'I_0' : 10,
            'omega':1/10
            }
    q = [0,0.2,0.4,0.6,0.8]
    a = np.linspace(0,0.8,10)
    for x in range(r):
        G = configuration_network(N,k0).build()
        model = stoch_mixed_tracing(G,quarantine_S_contacts = True)
        t,result = analysis(model,parameter).stoch_two_range_result('app_participation',a,'q',q,time)
        for i in a:
            for j in q:
                axs[0,0].plot(i, (result[i][j]['S'].max(axis = 0)+result[i][j]['Sa'].max(axis = 0))/N,'.', alpha = 0.3, color = colors[q.index(j)])
                axs[0,1].plot(i, (result[i][j]['E'].max(axis = 0)+result[i][j]['Ea'].max(axis = 0))/N,'.', alpha = 0.3,color = colors[q.index(j)])
                axs[0,2].plot(i, (result[i][j]['I_P'].max(axis = 0)+result[i][j]['I_Pa'].max(axis = 0))/N,'.', alpha = 0.3,color = colors[q.index(j)])
                axs[0,3].plot(i, (result[i][j]['I_S'].max(axis = 0)+result[i][j]['I_Sa'].max(axis = 0))/N,'.', alpha = 0.3,color = colors[q.index(j)])
                axs[1,0].plot(i, (result[i][j]['I_A'].max(axis = 0)+result[i][j]['I_Aa'].max(axis = 0))/N,'.', alpha = 0.3,color = colors[q.index(j)])
                axs[1,1].plot(i, (result[i][j]['T'].max(axis = 0)+result[i][j]['Ta'].max(axis = 0))/N,'.', alpha = 0.3,color = colors[q.index(j)])
                axs[1,2].plot(i, (result[i][j]['R'].max(axis = 0)+result[i][j]['Ra'].max(axis = 0)+result[i][j]['X'].max(axis = 0)+result[i][j]['Xa'].max(axis = 0))/N,'.', alpha = 0.3,color = colors[q.index(j)])
                axs[1,3].plot(i, (result[i][j]['Qa'].max(axis = 0))/N,'.', alpha = 0.3, color = colors[q.index(j)])
    for i in range(4):
        axs[1,i].set_xlabel(r'$a$')

    axs[0,0].set_ylabel(r'$S_{max}$')
    axs[0,1].set_ylabel(r'$E_{max}$')
    axs[0,2].set_ylabel(r'$I_{P_{max}}$')
    axs[0,3].set_ylabel(r'$I_{S_{max}}$')
    axs[1,0].set_ylabel(r'$I_{A_{max}}$')
    axs[1,1].set_ylabel(r'$T_{max}$')
    axs[1,2].set_ylabel(r'$R (t\rightarrow \infty)+X (t\rightarrow \infty)$')
    axs[1,3].set_ylabel(r'$Q_{max}$')

    lines = [Line2D([0], [0], color=colors[x], linewidth=3, linestyle='dotted') for x in range(len(q))]
    labels = [('q = ' + str(j)) for j in q]
    fig.legend(lines, labels)
    plt.show()

def tim():
    N = 40_000
    k0 = 19
    G = configuration_network(N,k0).build()
    time = 1000
    model = stoch_mixed_tracing(G,quarantine_S_contacts = True)
    parameter = {
            'R0': 2.5,
            'q': 0.5,
            'app_participation': 0.33,
            'chi':1/2.5,
            'recovery_rate' : 1/6,
            'alpha' : 1/2.5,
            'beta' : 1/2.5,
            #'number_of_contacts' : 6.3,
            'x':0.17,
            'y':0.1,
            'z':0.64,
            'I_0' : 400,
            'omega':1/10
            }

    model.set_parameters(parameter)
    t, result = model.compute(time)
    fig,ax = plt.subplots(1,2,figsize = (9,3),sharex = True, sharey = True)
    for i in ['E','I_P','I_S','I_A','T']:
    #for i in ['X','R','S']:
        ax[0].plot(t,result[i]/(N*(1-0.33)),label = i)
        ax[0].legend()
        ax[0].set_ylabel('fraction of individuals \n in not app-participants')
        ax[0].set_xlabel('time [d]')
    for i in ['Ea','I_Pa','I_Sa','I_Aa','Ta']:
    #for i in ['Xa','Ra','Sa']:
        ax[1].plot(t,result[i]/(N*(0.33)),label = i)
        ax[1].legend()
        ax[1].set_ylabel('fraction of individuals \n in app-participants')
        ax[1].set_xlabel('time [d]')
    plt.show()

#mixed_tracing_test()
stoch_mixed_tracing_test(10)
