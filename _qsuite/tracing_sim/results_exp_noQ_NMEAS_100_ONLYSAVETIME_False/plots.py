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
def relative_outbreak_reduction():
    sw_RX = []
    exp_RX = []
    for x in range(len(cf.q)):
        sw_RX.append(((sw[:,0,1,:,x,2] + sw[:,0,1,:,x,3]+sw[:,0,1,:,x,4] + sw[:,0,1,:,x,5]+sw[:,0,1,:,0,6])/200_000))
        exp_RX.append(((exp[:,0,1,:,x,2] + exp[:,0,1,:,x,3]+exp[:,0,1,:,x,4] + exp[:,0,1,:,x,5]+exp[:,0,1,:,0,6])/200_000))

    mean_sw_RX = np.mean(sw_RX,axis = 1)
    mean_exp_RX = np.mean(exp_RX,axis = 1)
    fig, ax = plt.subplots(2,2,figsize = (10,8))
    for x in range(6):

        ax[0,0].plot(mean_sw_RX[x], alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])
        ax[0,1].plot(mean_exp_RX[x], alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])
        ax[1,0].plot(((mean_sw_RX[x]/mean_sw_RX[x][0])-1)*100, alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])
        ax[1,1].plot(((mean_exp_RX[x]/mean_exp_RX[x][0])-1)*100, alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])

    positions = (0, 6, 12, 18, 24)
    xlabels = ('0',"0.25","0.5", "0.75",'1.0')

    for i in [0,1]:
        for j in [0,1]:
            ax[i,j].set_xticks(positions)
            ax[i,j].set_xticklabels(xlabels)
            ax[1,i].set_xlabel(r'a',**hfont)
            ax[0,0].set_ylabel(r'fraction of $R(t \rightarrow \infty)+X(t \rightarrow \infty)$ ',**hfont)
            ax[1,0].set_ylabel(r'reduction of $R(t \rightarrow \infty)+X(t \rightarrow \infty)$ [%]',**hfont)
            ax[0,i].set_ylim(0,0.8)
            ax[1,i].set_ylim(-85,5)

    ax[0,0].set_title(r'small-world network',**hfont)
    ax[0,1].set_title(r'exponential random network',**hfont)


    lines = [Line2D([0], [0], color = colors[x], marker = marker[x],linewidth=1.5, linestyle='-') for x in range(len(cf.q))]
    labels = ['q = 0','q = 0.1','q = 0.3','q = 0.5','q = 0.7','q = 0.9']
    ax[1,1].legend(lines, labels)

    fig.tight_layout()
    plt.savefig('rel_red_RX',dpi = 300)
    plt.show()
def relative_outbreak_reduction_lower_eff():
    sw_RX = []
    exp_RX = []
    for x in range(len(cf.q)):
        sw_RX.append(((sw[:,0,0,:,x,2] + sw[:,0,0,:,x,3]+sw[:,0,0,:,x,4] + sw[:,0,0,:,x,5]+sw[:,0,0,:,0,6])/200_000))
        exp_RX.append(((exp[:,0,0,:,x,2] + exp[:,0,0,:,x,3]+exp[:,0,0,:,x,4] + exp[:,0,0,:,x,5]+exp[:,0,0,:,0,6])/200_000))

    mean_sw_RX = np.mean(sw_RX,axis = 1)
    mean_exp_RX = np.mean(exp_RX,axis = 1)
    fig, ax = plt.subplots(2,2,figsize = (10,8))

    for x in range(6):
        ax[0,0].plot(mean_sw_RX[x], alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])
        ax[0,1].plot(mean_exp_RX[x], alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])
        ax[1,0].plot(((mean_sw_RX[x]/mean_sw_RX[x][0])-1)*100, alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])
        ax[1,1].plot(((mean_exp_RX[x]/mean_exp_RX[x][0])-1)*100, alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])

    positions = (0, 6, 12, 18, 24)
    xlabels = ('0',"0.25","0.5", "0.75",'1.0')
    for i in [0,1]:
        for j in [0,1]:
            ax[i,j].set_xticks(positions)
            ax[i,j].set_xticklabels(xlabels)
            ax[1,i].set_xlabel(r'a',**hfont)
            ax[0,0].set_ylabel(r'fraction of $R(t \rightarrow \infty)+X(t \rightarrow \infty)$ ',**hfont)
            ax[1,0].set_ylabel(r'reduction of $R(t \rightarrow \infty)+X(t \rightarrow \infty)$ [%]',**hfont)
            ax[0,i].set_ylim(0,0.8)
            ax[1,i].set_ylim(-85,5)

    ax[0,0].set_title(r'small-world network',**hfont)
    ax[0,1].set_title(r'exponential random network',**hfont)


    lines = [Line2D([0], [0], color = colors[x], marker = marker[x],linewidth=1.5, linestyle='-') for x in range(len(cf.q))]
    labels = ['q = 0','q = 0.1','q = 0.3','q = 0.5','q = 0.7','q = 0.9']
    ax[1,1].legend(lines, labels)

    fig.tight_layout()
    plt.savefig('rel_red_RX_lower_eff',dpi = 300)
    plt.show()
def darkfactor():
    expR = []
    expX = []
    expQ = []
    swR = []
    swX = []
    swQ = []
    for x in range(len(cf.q)):
        expR.append(exp[:,0,1,:,x,2] + exp[:,0,1,:,x,3])
        expQ.append(exp[:,0,1,:,x,6])
        expX.append(exp[:,0,1,:,x,4] + exp[:,0,1,:,x,5])
        swR.append(sw[:,0,1,:,x,2] + sw[:,0,1,:,x,3])
        swQ.append(sw[:,0,1,:,x,6])
        swX.append(sw[:,0,1,:,x,4] + sw[:,0,1,:,x,5])

    expR = np.array(expR)
    expX = np.array(expX)
    expQ = np.array(expQ)
    swR = np.array(swR)
    swX = np.array(swX)
    swQ = np.array(swQ)
    DFexp = (expR+expX+expQ)/(expX+expQ)
    DFsw = (swR+swX+swQ)/(swX+swQ)
    meanDFexp = np.mean(DFexp,axis = 1)
    meanDFsw = np.mean(DFsw,axis = 1)

    fig, ax = plt.subplots(1,2,figsize = (10,4),sharey = True)

    for x in range(6):
        ax[0].plot(meanDFsw[x], alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])
        ax[1].plot(meanDFexp[x], alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])


    positions = (0, 6, 12, 18, 24)
    xlabels = ('0',"0.25","0.5", "0.75",'1.0')
    for i in [0,1]:
        ax[i].set_xticks(positions)
        ax[i].set_xticklabels(xlabels)
        ax[i].set_xlabel(r'a',**hfont)
        ax[0].set_ylabel(r'DF',**hfont)

    ax[0].set_title(r'small-world network',**hfont)
    ax[1].set_title(r'exponential random network',**hfont)


    lines = [Line2D([0], [0], color = colors[x], marker = marker[x],linewidth=1.5, linestyle='-') for x in [1,2,3,4,5]]
    labels = ['q = 0.1','q = 0.3','q = 0.5','q = 0.7','q = 0.9']
    ax[1].legend(lines, labels)

    fig.tight_layout()
    #plt.savefig('darkfactor',dpi = 300)
    plt.show()
def darkfactor_lower_eff():
    expR = []
    expX = []
    expQ = []
    swR = []
    swX = []
    swQ = []
    for x in range(len(cf.q)):
        expR.append(exp[:,0,0,:,x,2] + exp[:,0,0,:,x,3])
        expQ.append(exp[:,0,0,:,x,6])
        expX.append(exp[:,0,0,:,x,4] + exp[:,0,0,:,x,5])
        swR.append(sw[:,0,0,:,x,2] + sw[:,0,0,:,x,3])
        swQ.append(sw[:,0,0,:,x,6])
        swX.append(sw[:,0,0,:,x,4] + sw[:,0,0,:,x,5])

    expR = np.array(expR)
    expX = np.array(expX)
    expQ = np.array(expQ)
    swR = np.array(swR)
    swX = np.array(swX)
    swQ = np.array(swQ)
    DFexp = (expR+expX+expQ)/(expX+expQ)
    DFsw = (swR+swX+swQ)/(swX+swQ)
    meanDFexp = np.mean(DFexp,axis = 1)
    meanDFsw = np.mean(DFsw,axis = 1)

    fig, ax = plt.subplots(1,2,figsize = (10,4),sharey = True)

    for x in range(6):
        ax[0].plot(meanDFsw[x], alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])
        ax[1].plot(meanDFexp[x], alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])

    positions = (0, 6, 12, 18, 24)
    xlabels = ('0',"0.25","0.5", "0.75",'1.0')
    for i in [0,1]:
        ax[i].set_xticks(positions)
        ax[i].set_xticklabels(xlabels)
        ax[i].set_xlabel(r'a',**hfont)
        ax[0].set_ylabel(r'DF',**hfont)

    ax[0].set_title(r'small-world network',**hfont)
    ax[1].set_title(r'exponential random network',**hfont)


    lines = [Line2D([0], [0], color = colors[x], marker = marker[x],linewidth=1.5, linestyle='-') for x in [1,2,3,4,5]]
    labels = ['q = 0.1','q = 0.3','q = 0.5','q = 0.7','q = 0.9']
    ax[1].legend(lines, labels)

    fig.tight_layout()
    plt.savefig('darkfactor_lower_eff',dpi = 300)
    plt.show()

def darkfactor_alternative():
    expR = []
    expX = []
    expQ = []
    swR = []
    swX = []
    swQ = []
    for x in range(len(cf.q)):
        expR.append(exp[:,0,1,:,x,2] + exp[:,0,1,:,x,3])
        expQ.append(exp[:,0,1,:,x,6])
        expX.append(exp[:,0,1,:,x,4] + exp[:,0,1,:,x,5])
        swR.append(sw[:,0,1,:,x,2] + sw[:,0,1,:,x,3])
        swQ.append(sw[:,0,1,:,x,6])
        swX.append(sw[:,0,1,:,x,4] + sw[:,0,1,:,x,5])

    expR = np.array(expR)
    expX = np.array(expX)
    expQ = np.array(expQ)
    swR = np.array(swR)
    swX = np.array(swX)
    swQ = np.array(swQ)
    DFexp = (expR+expX+expQ)/(expX)
    DFsw = (swR+swX+swQ)/(swX)
    meanDFexp = np.mean(DFexp,axis = 1)
    meanDFsw = np.mean(DFsw,axis = 1)

    fig, ax = plt.subplots(1,2,figsize = (10,4),sharey = True)

    for x in range(6):
        ax[0].plot(meanDFsw[x], alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])
        ax[1].plot(meanDFexp[x], alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])


    positions = (0, 6, 12, 18, 24)
    xlabels = ('0',"0.25","0.5", "0.75",'1.0')
    for i in [0,1]:
        ax[i].set_xticks(positions)
        ax[i].set_xticklabels(xlabels)
        ax[i].set_xlabel(r'a',**hfont)
        ax[0].set_ylabel(r'DF',**hfont)

    ax[0].set_title(r'small-world network',**hfont)
    ax[1].set_title(r'exponential random network',**hfont)


    lines = [Line2D([0], [0], color = colors[x], marker = marker[x],linewidth=1.5, linestyle='-') for x in [1,2,3,4,5]]
    labels = ['q = 0.1','q = 0.3','q = 0.5','q = 0.7','q = 0.9']
    ax[1].legend(lines, labels)

    fig.tight_layout()
    #plt.savefig('darkfactor',dpi = 300)
    plt.show()
darkfactor_alternative()
