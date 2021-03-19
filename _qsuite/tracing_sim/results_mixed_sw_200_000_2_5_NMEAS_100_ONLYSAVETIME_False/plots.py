import pickle
import matplotlib.pyplot as plt
hfont = {'fontname':'Helvetica'}
plt.rcParams.update({'font.size': 13})
import numpy as np
from matplotlib.lines import Line2D
import gzip
import qsuite_config as cf
data = pickle.load(gzip.open('/Users/angeliqueburdinski/Desktop/Arbeit/tracing_sim/results_mixed_sw_200_000_2_5_NMEAS_100_ONLYSAVETIME_False/results.p.gz','rb'))
data = np.array(data)
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
def sw_plot():
    fig, axs = plt.subplots(2, 3,figsize=(9,6), sharex=True, sharey=True)
    for i in range(len(data)):
        axs[0,0].plot(cf.a, data[i,:,0,1],'o', alpha = 0.1, color = colors[0])
        axs[0,0].set_title("q = 0",**hfont)
        axs[0,0].set_ylabel(r'$R (t\rightarrow \infty)+X (t\rightarrow \infty)$',**hfont)
    for x in range(len(data)):
        axs[0,1].plot(cf.a,data[x,:,1,1],'o', alpha = 0.1, color = colors[1])
        axs[0,1].set_title("q = 0.01",**hfont)
    for x in range(len(data)):
        axs[0,2].plot(cf.a,data[x,:,2,1],'o', alpha = 0.1, color = colors[2])
        axs[0,2].set_title("q = 0.1",**hfont)
    for x in range(len(data)):
        axs[1,0].plot(cf.a,data[x,:,3,1],'o', alpha = 0.1, color = colors[3])
        axs[1,0].set_title("q = 0.3",**hfont)
        axs[1,0].set_xlabel("a",**hfont)
        axs[1,0].set_ylabel(r'$R (t\rightarrow \infty)+X (t\rightarrow \infty)$',**hfont)
    for x in range(len(data)):
        axs[1,1].plot(cf.a,data[x,:,4,1],'o', alpha = 0.1, color = colors[4])
        axs[1,1].set_title("q = 0.5",**hfont)
        axs[1,1].set_xlabel("a",**hfont)
    for x in range(len(data)):
        axs[1,2].plot(cf.a,data[x,:,5,1],'o', alpha = 0.1, color = colors[5])
        axs[1,2].set_title("q = 0.7",**hfont)
        axs[1,2].set_xlabel("a",**hfont)
    plt.tight_layout()
    plt.savefig(str(int(cf.parameter['R0']))+'_SW_RX_'+str(cf.N))
    plt.show()


    fig, axs = plt.subplots(2, 3,figsize=(9,6), sharex=True, sharey=True)
    for i in range(len(data)):
        axs[0,0].plot(cf.a, data[i,:,0,0],'o', alpha = 0.1, color = colors[0])
        axs[0,0].set_title("q = 0",**hfont)
        axs[0,0].set_ylabel(r'$I_{S_{max}}$',**hfont)
    for x in range(len(data)):
        axs[0,1].plot(cf.a,data[x,:,1,0],'o', alpha = 0.1, color = colors[1])
        axs[0,1].set_title("q = 0.01",**hfont)
    for x in range(len(data)):
        axs[0,2].plot(cf.a,data[x,:,2,0],'o', alpha = 0.1, color = colors[2])
        axs[0,2].set_title("q = 0.1",**hfont)
    for x in range(len(data)):
        axs[1,0].plot(cf.a,data[x,:,3,0],'o', alpha = 0.1, color = colors[3])
        axs[1,0].set_title("q = 0.3",**hfont)
        axs[1,0].set_xlabel("a",**hfont)
        axs[1,0].set_ylabel(r'$I_{S_{max}}$',**hfont)
    for x in range(len(data)):
        axs[1,1].plot(cf.a,data[x,:,4,0],'o', alpha = 0.1, color = colors[4])
        axs[1,1].set_title("q = 0.5",**hfont)
        axs[1,1].set_xlabel("a",**hfont)
    for x in range(len(data)):
        axs[1,2].plot(cf.a,data[x,:,5,0],'o', alpha = 0.1, color = colors[5])
        axs[1,2].set_title("q = 0.7",**hfont)
        axs[1,2].set_xlabel("a",**hfont)
    plt.tight_layout()
    plt.savefig(str(int(cf.parameter['R0']))+'_SW_Is_'+str(cf.N))
    plt.show()



def config_plot():
    #colors = colors[0:len(cf.q)]
    for i in range(len(data)):
        for x in [0,2,3,4,5,6]:
            plt.plot(data[i,:,x,0],'o', alpha = 0.1, color = colors[x])

    positions = (0, 5, 10, 15, 20, 25)
    labels = ('0',"0.2","0.4", "0.6", "0.8",'1.0')
    plt.xticks(positions, labels)
    lines = [Line2D([0], [0], color = colors[x], linewidth=3, linestyle='dotted') for x in [0,2,3,4,5,6]]
    labels = ['q = 0','q = 0.1','q = 0.3','q = 0.5','q = 0.7','q = 0.9']
    plt.legend(lines, labels)
    plt.xlabel('a',**hfont)
    plt.ylabel(' $I_{S_{max}}$',**hfont)

    plt.ylim(0,200_000)
    plt.show()

    for i in range(len(data)):
        for x in [0,2,3,4,5,6]:
            plt.plot(data[i,:,x,1],'o', alpha = 0.1, color = colors[x])

    positions = (0, 5, 10, 15, 20, 25)
    labels = ('0',"0.2","0.4", "0.6", "0.8",'1.0')
    plt.xticks(positions, labels)
    lines = [Line2D([0], [0], color = colors[x], linewidth=3, linestyle='dotted') for x in [0,2,3,4,5,6]]
    #labels = ['q = '+str(i) for i in cf.q]
    labels = ['q = 0','q = 0.1','q = 0.3','q = 0.5','q = 0.7','q = 0.9']
    plt.legend(lines, labels)
    plt.xlabel('a',**hfont)
    plt.ylabel(r'$R (t\rightarrow \infty)+X (t\rightarrow \infty)$',**hfont)
    plt.ylim(0,200_000)
    plt.show()

config_plot()
