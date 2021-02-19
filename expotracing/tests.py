from extendedmodel import (first_generation_tracing,next_generation_tracing,mixed_tracing,stoch_mixed_tracing)
from tools import (analysis,configuration_network)
from plots import plot
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from smallworld import get_smallworld_graph
from labellines import labelLine, labelLines
colors = [
'dimgrey',
'lightcoral',
'skyblue',
'indianred',
'darkcyan',
'maroon',
'darkgoldenrod',
'navy',
'mediumvioletred',
'darkseagreen',
'crimson'
]
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
    comps = ['E','I_P','I_S','I_A','T']
    comps_a = ['Ea','I_Pa','I_Sa','I_Aa','Ta']
    for i in comps:
        ax[0].plot(t,result[i]/(N*(1-0.33)),label = i,color = colors[comps.index(i)])
        ax[0].legend()
        ax[0].set_ylabel('fraction of individuals \n in not app-participants')
        ax[0].set_xlabel('time [d]')
    for i in comps_a:
        ax[1].plot(t,result[i]/(N*(0.33)),label = i,color = colors[comps_a.index(i)])
        ax[1].legend()
        ax[1].set_ylabel('fraction of individuals \n in app-participants')
        ax[1].set_xlabel('time [d]')
    plt.show()

def confplotnew(meas):
    N= 10000
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
            'number_of_contacts' : 6.3,
            'x':0.17,
            'y':0.1,
            'z':0.64,
            'I_0' : 10,
            'omega':1/10
            }

    q = [0,0.2,0.4,0.6,0.8]
    a = np.linspace(0,0.8,5)

    qresult = []
    isresult = []
    iaresult = []
    ipresult =[]
    sqresult = []

    for w in range(meas):
        G = configuration_network(N,k0).build()
        model = stoch_mixed_tracing(G,quarantine_S_contacts = True)
        t,result = analysis(model,parameter).stoch_two_range_result('app_participation',a,'q',q,time)

        q_res = []
        is_res = []
        ia_res = []
        ip_res = []
        sq_res = []
        for j in q:
            q_res.append([(result[i][j]['R'].max(axis = 0)+result[i][j]['Ra'].max(axis = 0)+result[i][j]['X'].max(axis = 0)+result[i][j]['Xa'].max(axis = 0))/N for i in a])
            is_res.append([(result[i][j]['I_S'].max(axis = 0)+result[i][j]['I_Sa'].max(axis = 0))/N for i in a])
            ia_res.append([(result[i][j]['I_A'].max(axis = 0)+result[i][j]['I_Aa'].max(axis = 0))/N for i in a])
            ip_res.append([(result[i][j]['I_P'].max(axis = 0)+result[i][j]['I_Pa'].max(axis = 0))/N for i in a])
            sq_res.append([(result[i][j]['Qa'].max(axis = 0))/N for i in a])

        qresult.append(q_res)
        isresult.append(is_res)
        iaresult.append(ia_res)
        ipresult.append(ip_res)
        sqresult.append(sq_res)

    for x in qresult:
        for j in x:
            plt.plot(a, j, '-',color = colors[x.index(j)],alpha = 0.5)

    plt.xlabel(r'$a$')
    plt.ylabel(r'$R (t\rightarrow \infty)+X (t\rightarrow \infty)$')
    labels = ['q = '+str(i) for i in q]
    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in colors]
    plt.legend(lines,labels, loc='best')
    plt.show()

    for x in isresult:
        for j in x:
            plt.plot(a, j, '-',color = colors[x.index(j)],alpha = 0.5)

    plt.xlabel(r'$a$')
    plt.ylabel(r'$I_{S_{max}}$')
    labels = ['q = '+str(i) for i in q]
    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in colors]
    plt.legend(lines,labels, loc='best')
    plt.show()

    for x in ipresult:
        for j in x:
            plt.plot(a, j, '-',color = colors[x.index(j)],alpha = 0.5)

    plt.xlabel(r'$a$')
    plt.ylabel(r'$I_{P_{max}}$')
    labels = ['q = '+str(i) for i in q]
    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in colors]
    plt.legend(lines,labels, loc='best')
    plt.show()

    for x in iaresult:
        for j in x:
            plt.plot(a, j, '-',color = colors[x.index(j)],alpha = 0.5)

    plt.xlabel(r'$a$')
    plt.ylabel(r'$I_{A_{max}}$')
    labels = ['q = '+str(i) for i in q]
    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in colors]
    plt.legend(lines,labels, loc='best')
    plt.show()

    for x in sqresult:
        for j in x:
            plt.plot(a, j, '-',color = colors[x.index(j)],alpha = 0.5)

    plt.xlabel(r'$a$')
    plt.ylabel(r'$Q_{max}$')
    labels = ['q = '+str(i) for i in q]
    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in colors]
    plt.legend(lines,labels, loc='best')
    plt.show()

def mixplotnew():
    N = 10_000
    t = np.linspace(0,500,500)
    model = mixed_tracing(N,quarantine_S_contacts = True)
    parameter = {
            'R0': 2.5,
            'q': 0.5,
            'app_participation': 0.33,
            #'chi':1,
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
    z = [0,0.1,0.3,0.5,0.7,0.9,1]
    q = np.linspace(0,0.9,50)


    result = analysis(model,parameter).two_range_result('q',q,'z',z,t)
    q_res = []
    for j in z:
        q_res.append([(result[i][j]['R'].max(axis = 0)+result[i][j]['X'].max(axis = 0))/N for i in q])

    for x in q_res:
        plt.plot(q, x, '-',color = colors[q_res.index(x)],label= 'z = '+str(z[q_res.index(x)]))


    plt.xlabel(r'$a$')
    plt.ylabel(r'$R (t\rightarrow \infty)+X (t\rightarrow \infty)$')
    labelLines(plt.gca().get_lines(),zorder=2.5)
    #plt.savefig('RX')
    plt.show()

    ip_res = []
    for j in z:
        ip_res.append([(result[i][j]['I_P'].max(axis = 0))/N for i in q])

    for x in ip_res:
        plt.plot(q, x, '-',color = colors[ip_res.index(x)],label= 'z = '+str(z[ip_res.index(x)]))

    plt.xlabel(r'$a$')
    plt.ylabel(r'$I_{P_{max}}$')
    labelLines(plt.gca().get_lines(),zorder=2.5)

    #plt.savefig('I_Pmax')
    plt.show()

    is_res = []
    for j in z:
        is_res.append([(result[i][j]['I_S'].max(axis = 0))/N for i in q])

    for x in is_res:
        plt.plot(q, x, '-',color = colors[is_res.index(x)],label= 'z = '+str(z[is_res.index(x)]))

    plt.xlabel(r'$q$')
    plt.ylabel(r'$I_{S_{max}}$')
    labelLines(plt.gca().get_lines(),zorder=2.5)

    #plt.savefig('I_Smax')
    plt.show()

    ia_res = []
    for j in z:
        ia_res.append([(result[i][j]['I_A'].max(axis = 0))/N for i in q])

    for x in ia_res:
        plt.plot(q, x, '-',color = colors[ia_res.index(x)],label= 'z = '+str(z[ia_res.index(x)]))

    plt.xlabel(r'$a$')
    plt.ylabel(r'$I_{A_{max}}$')
    labelLines(plt.gca().get_lines(),zorder=2.5)

    #plt.savefig('I_Amax')
    plt.show()


    Q_res = []
    for j in z:
        Q_res.append([(result[i][j]['Q'].max(axis = 0))/N for i in q])

    for x in Q_res:
        plt.plot(q, x, '-',color = colors[Q_res.index(x)],label= 'z = '+str(z[q_res.index(x)]))

    plt.xlabel(r'$a$')
    plt.ylabel(r'$Q_{max}$')
    labelLines(plt.gca().get_lines(),zorder=2.5)
    #plt.savefig('Q')
    plt.show()

def swstoch(meas):
    N= 10_000
    #k0 = 20
    time = 1000
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

    q = [0,0.2,0.4]
    a = np.linspace(0,0.8,20)

    qresult = []
    isresult = []
    iaresult = []
    ipresult =[]
    sqresult = []
    #meas = 20
    k_over_2 = 10
    beta = 10e-6
    for w in range(meas):
        G = get_smallworld_graph(N,k_over_2,beta)
        #G = nx.barabasi_albert_graph(N,10)
        #G = configuration_network(N,k0).build()
        model = stoch_mixed_tracing(G,quarantine_S_contacts = True)
        t,result = analysis(model,parameter).stoch_two_range_result('app_participation',a,'q',q,time)

        q_res = []
        is_res = []
        ia_res = []
        ip_res = []
        sq_res = []
        for j in q:
            q_res.append([(result[i][j]['R'].max(axis = 0)+result[i][j]['Ra'].max(axis = 0)+result[i][j]['X'].max(axis = 0)+result[i][j]['Xa'].max(axis = 0))/N for i in a])
            is_res.append([(result[i][j]['I_S'].max(axis = 0)+result[i][j]['I_Sa'].max(axis = 0))/N for i in a])
            ia_res.append([(result[i][j]['I_A'].max(axis = 0)+result[i][j]['I_Aa'].max(axis = 0))/N for i in a])
            ip_res.append([(result[i][j]['I_P'].max(axis = 0)+result[i][j]['I_Pa'].max(axis = 0))/N for i in a])
            sq_res.append([(result[i][j]['Qa'].max(axis = 0))/N for i in a])

        qresult.append(q_res)
        isresult.append(is_res)
        iaresult.append(ia_res)
        ipresult.append(ip_res)
        sqresult.append(sq_res)

    for x in qresult:
        for j in x:
            plt.plot(a, j, '.',color = colors[x.index(j)],alpha = 0.5)

    plt.xlabel(r'$a$')
    plt.ylabel(r'$R (t\rightarrow \infty)+X (t\rightarrow \infty)$')
    labels = ['q = '+str(i) for i in q]
    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='dotted') for c in colors]
    plt.legend(lines,labels, loc='best')
    plt.savefig('SW_RX')
    plt.show()

    for x in isresult:
        for j in x:
            plt.plot(a, j, '.',color = colors[x.index(j)],alpha = 0.5)

    plt.xlabel(r'$a$')
    plt.ylabel(r'$I_{S_{max}}$')
    labels = ['q = '+str(i) for i in q]
    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='dotted') for c in colors]
    plt.legend(lines,labels, loc='best')
    plt.savefig('SW_IS')
    plt.show()

    for x in ipresult:
        for j in x:
            plt.plot(a, j, '.',color = colors[x.index(j)],alpha = 0.5)

    plt.xlabel(r'$a$')
    plt.ylabel(r'$I_{P_{max}}$')
    labels = ['q = '+str(i) for i in q]
    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='dotted') for c in colors]
    plt.legend(lines,labels, loc='best')
    plt.savefig('SW_IP')
    plt.show()

    for x in iaresult:
        for j in x:
            plt.plot(a, j, '.',color = colors[x.index(j)],alpha = 0.5)

    plt.xlabel(r'$a$')
    plt.ylabel(r'$I_{A_{max}}$')
    labels = ['q = '+str(i) for i in q]
    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='dotted') for c in colors]
    plt.legend(lines,labels, loc='best')
    plt.savefig('SW_IA')
    plt.show()

    for x in sqresult:
        for j in x:
            plt.plot(a, j, '.',color = colors[x.index(j)],alpha = 0.5)

    plt.xlabel(r'$a$')
    plt.ylabel(r'$Q_{max}$')
    labels = ['q = '+str(i) for i in q]
    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='dotted') for c in colors]
    plt.legend(lines,labels, loc='best')
    plt.savefig('SW_Q')
    plt.show()


N = 10_000
t = np.linspace(0,150,600)
model = mixed_tracing(N,quarantine_S_contacts = True)
parameter = {
        'R0': 2.5,
        'q': 0.5,
        'app_participation': 0.33,
        #'chi':1,
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
parameter.update({'y':1})
parameter.update({'z':1})
#parameter.update({'chi':1})
parameter.update({'app_participation':0.8})

#I0 = [1,10,100,1000]
q = np.linspace(0,0.9,10)
plot(model,parameter).range_plot('q',q,['I_S','I_P'],t)
