import pickle
import matplotlib.pyplot as plt

hfont = {'fontname':'Helvetica'}
plt.rcParams.update({'font.size': 12})
import numpy as np
from matplotlib.lines import Line2D
import gzip
import qsuite_config as cf
sw = pickle.load(gzip.open('/Users/angeliqueburdinski/expotracing/_qsuite/tracing_sim/results_sw_NMEAS_100_ONLYSAVETIME_False/results.p.gz','rb'))
sw = np.array(sw)
exp = pickle.load(gzip.open('/Users/angeliqueburdinski/expotracing/_qsuite/tracing_sim/results_exp_NMEAS_100_ONLYSAVETIME_False/results.p.gz','rb'))
exp = np.array(exp)
sw_noQ = pickle.load(gzip.open('/Users/angeliqueburdinski/expotracing/_qsuite/tracing_sim/results_sw_noQ_NMEAS_100_ONLYSAVETIME_False/results.p.gz','rb'))
sw_noQ = np.array(sw_noQ)
exp_noQ = pickle.load(gzip.open('/Users/angeliqueburdinski/expotracing/_qsuite/tracing_sim/results_exp_noQ_NMEAS_100_ONLYSAVETIME_False/results.p.gz','rb'))
exp_noQ = np.array(exp_noQ)
marker = ['o','v','x','d','p','<']
colors = [
        'k',
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

def Fig1():
    expR = []
    expX = []
    expC = []
    for x in range(len(cf.q)):
        expR.append(exp[:,0,1,:,x,2] + exp[:,0,1,:,x,3])
        expC.append(exp[:,0,1,:,x,6])
        expX.append(exp[:,0,1,:,x,4] + exp[:,0,1,:,x,5])

    expR = np.array(expR)
    expX = np.array(expX)
    expC = np.array(expC)

    omega = expR+expX+expC
    DF = (omega)/expX

    mean_DF = np.mean(DF,axis = 1)
    mean_omega = np.mean(omega,axis = 1)

    fig, ax = plt.subplots(1,3,figsize = (16,4))

    for x in range(6):
        ax[0].plot(mean_omega[x]/200_000, alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])
        ax[1].plot(((mean_omega[x]/mean_omega[x][0])-1)*100, alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])
        ax[2].plot(mean_DF[x], alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])

    positions = (0, 6, 12, 18, 24)
    xlabels = ('0',"0.25","0.5", "0.75",'1.0')

    for i in [0,1,2]:
        ax[i].set_xticks(positions)
        ax[i].set_xticklabels(xlabels)
        ax[i].set_xlabel(r'a',**hfont)
        ax[0].set_ylim(0.2,0.7)
        ax[1].set_ylim(-55,5)
        ax[2].set_ylim(0,13)
        ax[0].set_ylabel(r'$\Omega$ ',**hfont)
        ax[1].set_ylabel(r'reduction of $\Omega$ [%] ',**hfont)
        ax[2].set_ylabel(r'dark factor',**hfont)

    lines = [Line2D([0], [0], color = colors[x], marker = marker[x],linewidth=1.5, linestyle='-') for x in range(len(cf.q))]
    labels = ['q = 0','q = 0.1','q = 0.3','q = 0.5','q = 0.7','q = 0.9']
    ax[0].legend(lines, labels)
    plt.savefig('Fig1',dpi = 300)
    plt.show()

def Fig2():
    exp_RX = []
    exp_low_eff_RX = []
    exp_noQ_RX = []

    for x in range(len(cf.q)):
        exp_RX.append(((exp[:,0,1,:,x,2] + exp[:,0,1,:,x,3]+exp[:,0,1,:,x,4] + exp[:,0,1,:,x,5]+exp[:,0,1,:,x,6])/200_000))
        exp_low_eff_RX.append(((exp[:,0,0,:,x,2] + exp[:,0,0,:,x,3]+exp[:,0,0,:,x,4] + exp[:,0,0,:,x,5]+exp[:,0,0,:,x,6])/200_000))
        exp_noQ_RX.append(((exp_noQ[:,0,0,:,x,2] + exp_noQ[:,0,0,:,x,3]+exp_noQ[:,0,0,:,x,4] + exp_noQ[:,0,0,:,x,5]+exp_noQ[:,0,0,:,x,6])/200_000))

    mean_exp_RX = np.mean(exp_RX,axis = 1)
    mean_exp_low_eff_RX = np.mean(exp_low_eff_RX,axis = 1)
    mean_exp_noQ_RX = np.mean(exp_noQ_RX,axis = 1)

    fig, ax = plt.subplots(2,3,figsize = (15,8))

    for x in range(6):
        ax[0,0].plot(mean_exp_RX[x], alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])
        ax[1,0].plot(((mean_exp_RX[x]/mean_exp_RX[x][0])-1)*100, alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])
        ax[0,1].plot(mean_exp_noQ_RX[x], alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])
        ax[1,1].plot(((mean_exp_noQ_RX[x]/mean_exp_noQ_RX[x][0])-1)*100, alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])
        ax[0,2].plot(mean_exp_low_eff_RX[x], alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])
        ax[1,2].plot(((mean_exp_low_eff_RX[x]/mean_exp_low_eff_RX[x][0])-1)*100, alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])


    positions = (0, 6, 12, 18, 24)
    xlabels = ('0',"0.25","0.5", "0.75",'1.0')

    for i in [0,1,2]:
        for j in [0,1]:
            ax[j,i].set_xticks(positions)
            ax[j,i].set_xticklabels(xlabels)
            ax[j,i].set_xlabel(r'a',**hfont)
            ax[0,i].set_ylim(0.2,0.7)
            ax[1,i].set_ylim(-55,5)
            ax[0,0].set_ylabel(r'$\Omega$ ',**hfont)
            ax[1,0].set_ylabel(r'reduction of $\Omega$ [%] ',**hfont)

    lines = [Line2D([0], [0], color = colors[x], marker = marker[x],linewidth=1.5, linestyle='-') for x in range(len(cf.q))]
    labels = ['q = 0','q = 0.1','q = 0.3','q = 0.5','q = 0.7','q = 0.9']

    ax[0,0].legend(lines, labels)

    plt.savefig('Fig2',dpi = 300)
    plt.show()
def Fig3():
    swR = []
    swX = []
    swC = []
    for x in range(len(cf.q)):
        swR.append(sw[:,0,1,:,x,2] + sw[:,0,1,:,x,3])
        swC.append(sw[:,0,1,:,x,6])
        swX.append(sw[:,0,1,:,x,4] + sw[:,0,1,:,x,5])

    swR = np.array(swR)
    swX = np.array(swX)
    swC = np.array(swC)

    omega = swR+swX+swC
    DF = (omega)/swX

    mean_DF = np.mean(DF,axis = 1)
    mean_omega = np.mean(omega,axis = 1)

    fig, ax = plt.subplots(1,3,figsize = (16,4))

    for x in range(6):
        ax[0].plot(mean_omega[x]/200_000, alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])
        ax[1].plot(((mean_omega[x]/mean_omega[x][0])-1)*100, alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])
        ax[2].plot(mean_DF[x], alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])

    positions = (0, 6, 12, 18, 24)
    xlabels = ('0',"0.25","0.5", "0.75",'1.0')

    for i in [0,1,2]:
        ax[i].set_xticks(positions)
        ax[i].set_xticklabels(xlabels)
        ax[i].set_xlabel(r'a',**hfont)
        ax[0].set_ylim(0,0.75)
        ax[1].set_ylim(-75,5)
        ax[2].set_ylim(0,13)
        ax[0].set_ylabel(r'$\Omega$ ',**hfont)
        ax[1].set_ylabel(r'reduction of $\Omega$ [%] ',**hfont)
        ax[2].set_ylabel(r'dark factor',**hfont)

    lines = [Line2D([0], [0], color = colors[x], marker = marker[x],linewidth=1.5, linestyle='-') for x in range(len(cf.q))]
    labels = ['q = 0','q = 0.1','q = 0.3','q = 0.5','q = 0.7','q = 0.9']
    ax[0].legend(lines, labels)
    plt.savefig('Fig3',dpi = 300)
    plt.show()
def Fig4():
    sw_RX = []
    sw_low_eff_RX = []
    sw_noQ_RX = []

    for x in range(len(cf.q)):
        sw_RX.append(((sw[:,0,1,:,x,2] + sw[:,0,1,:,x,3]+sw[:,0,1,:,x,4] + sw[:,0,1,:,x,5]+sw[:,0,1,:,x,6])/200_000))
        sw_low_eff_RX.append(((sw[:,0,0,:,x,2] + sw[:,0,0,:,x,3]+sw[:,0,0,:,x,4] + sw[:,0,0,:,x,5]+sw[:,0,0,:,x,6])/200_000))
        sw_noQ_RX.append(((sw_noQ[:,0,0,:,x,2] + sw_noQ[:,0,0,:,x,3]+sw_noQ[:,0,0,:,x,4] + sw_noQ[:,0,0,:,x,5]+sw_noQ[:,0,0,:,x,6])/200_000))

    mean_sw_RX = np.mean(sw_RX,axis = 1)
    mean_sw_low_eff_RX = np.mean(sw_low_eff_RX,axis = 1)
    mean_sw_noQ_RX = np.mean(sw_noQ_RX,axis = 1)

    fig, ax = plt.subplots(2,3,figsize = (15,8))

    for x in range(6):
        ax[0,0].plot(mean_sw_RX[x], alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])
        ax[1,0].plot(((mean_sw_RX[x]/mean_sw_RX[x][0])-1)*100, alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])
        ax[0,1].plot(mean_sw_noQ_RX[x], alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])
        ax[1,1].plot(((mean_sw_noQ_RX[x]/mean_sw_noQ_RX[x][0])-1)*100, alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])
        ax[0,2].plot(mean_sw_low_eff_RX[x], alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])
        ax[1,2].plot(((mean_sw_low_eff_RX[x]/mean_sw_low_eff_RX[x][0])-1)*100, alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])


    positions = (0, 6, 12, 18, 24)
    xlabels = ('0',"0.25","0.5", "0.75",'1.0')

    for i in [0,1,2]:
        for j in [0,1]:
            ax[j,i].set_xticks(positions)
            ax[j,i].set_xticklabels(xlabels)
            ax[j,i].set_xlabel(r'a',**hfont)
            ax[0,i].set_ylim(0,0.75)
            ax[1,i].set_ylim(-75,5)
            ax[0,0].set_ylabel(r'$\Omega$ ',**hfont)
            ax[1,0].set_ylabel(r'reduction of $\Omega$ [%] ',**hfont)

    lines = [Line2D([0], [0], color = colors[x], marker = marker[x],linewidth=1.5, linestyle='-') for x in range(len(cf.q))]
    labels = ['q = 0','q = 0.1','q = 0.3','q = 0.5','q = 0.7','q = 0.9']

    ax[1,2].legend(lines, labels)

    plt.savefig('Fig4',dpi = 300)
    plt.show()
Fig3()
