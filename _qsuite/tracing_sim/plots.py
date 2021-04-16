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
lockdown = pickle.load(gzip.open('/Users/angeliqueburdinski/expotracing/_qsuite/tracing_sim/results_lockdown_NMEAS_100_ONLYSAVETIME_False/results.p.gz','rb'))
lockdown = np.array(lockdown)
no_lockdown = pickle.load(gzip.open('/Users/angeliqueburdinski/expotracing/_qsuite/tracing_sim/results_nolockdown_NMEAS_100_ONLYSAVETIME_False/results.p.gz','rb'))
no_lockdown = np.array(no_lockdown)
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
    omega_sw = np.array([(sw[:,0,1,:,x,2]+sw[:,0,1,:,x,3]+sw[:,0,1,:,x,4] + sw[:,0,1,:,x,5]+sw[:,0,1,:,x,6]) for x in range(len(cf.q))])
    DF_sw = (omega_sw)/np.array([(sw[:,0,1,:,x,4] + sw[:,0,1,:,x,5]) for x in range(len(cf.q))])
    omega_exp = np.array([(exp[:,0,1,:,x,2] + exp[:,0,1,:,x,3]+exp[:,0,1,:,x,4] + exp[:,0,1,:,x,5]+exp[:,0,1,:,x,6])for x in range(len(cf.q))])
    DF_exp = (omega_exp)/np.array([(exp[:,0,1,:,x,4] + exp[:,0,1,:,x,5]) for x in range(len(cf.q))])

    mean_DFsw = np.mean(DF_sw,axis = 1)
    mean_omegasw = np.mean(omega_sw,axis = 1)/200_000
    mean_DFexp = np.mean(DF_exp,axis = 1)
    mean_omegaexp = np.mean(omega_exp,axis = 1)/200_000

    a = [0,7,12,19]
    fig, ax = plt.subplots(1,2,figsize = (14,5))

    for x in a:
        ax[0].bar(0-0.2,          mean_omegaexp[0][0], alpha = 1, color = colors[0], lw = 1.5, width = 0.2)
        ax[0].bar(a.index(x),     mean_omegaexp[2][x], alpha = 1, color = colors[1], lw = 1.5, width = 0.2)
        ax[0].bar(a.index(x)+0.2, mean_omegaexp[3][x], alpha = 1, color = colors[2], lw = 1.5, width = 0.2)
        ax[1].bar(0-0.2,          mean_omegasw[0][0],  alpha = 1, color = colors[0], lw = 1.5, width = 0.2)
        ax[1].bar(a.index(x),     mean_omegasw[2][x],  alpha = 1, color = colors[1], lw = 1.5, width = 0.2)
        ax[1].bar(a.index(x)+0.2, mean_omegasw[3][x],  alpha = 1, color = colors[2], lw = 1.5, width = 0.2)


    positions = (0.1, 1.1, 2.1, 3.1)
    xlabels = ("0", "30", "50", "80")

    for i in [0,1]:
        ax[i].set_xticks(positions)
        ax[i].set_xticklabels(xlabels)
        ax[i].set_xlabel(r'app participation $a$ [%]',**hfont)
        ax[i].set_ylim(0,0.8)
        ax[i].set_ylabel(r'outbreak size $\Omega$ ',**hfont)

    lines = [Line2D([0], [0], color = colors[x],linewidth=4, linestyle='-') for x in [0,1,2]]
    ylabels_0 = ['no testing']+[r'$DF_0$ = '+str(round(mean_DFexp[x][0],1)) for x in [2,3]]
    ylabels_1 = ['no testing']+[r'$DF_0$ = '+str(round(mean_DFsw[x][0],1)) for x in [2,3]]

    ax[0].set_title('random exponential')
    ax[1].set_title('small world')
    ax[0].legend(lines, ylabels_0)
    ax[1].legend(lines, ylabels_1)
    #plt.savefig('Fig1',dpi = 300)
    plt.show()

def Fig2():
    sw_RX =         [(sw[:,0,1,:,x,2] + sw[:,0,1,:,x,3]+sw[:,0,1,:,x,4] + sw[:,0,1,:,x,5]+sw[:,0,1,:,x,6]) for x in range(len(cf.q))]
    sw_low_eff_RX = [(sw[:,0,0,:,x,2] + sw[:,0,0,:,x,3]+sw[:,0,0,:,x,4] + sw[:,0,0,:,x,5]+sw[:,0,0,:,x,6]) for x in range(len(cf.q))]
    sw_noQ_RX =     [(sw_noQ[:,0,0,:,x,2] + sw_noQ[:,0,0,:,x,3]+sw_noQ[:,0,0,:,x,4] + sw_noQ[:,0,0,:,x,5]+sw_noQ[:,0,0,:,x,6]) for x in range(len(cf.q)) ]

    exp_RX =            [(exp[:,0,1,:,x,2] + exp[:,0,1,:,x,3]+exp[:,0,1,:,x,4] + exp[:,0,1,:,x,5]+exp[:,0,1,:,x,6]) for x in range(len(cf.q))]
    exp_low_eff_RX =    [(exp[:,0,0,:,x,2] + exp[:,0,0,:,x,3]+exp[:,0,0,:,x,4] + exp[:,0,0,:,x,5]+exp[:,0,0,:,x,6]) for x in range(len(cf.q))]
    exp_noQ_RX =        [(exp_noQ[:,0,0,:,x,2] + exp_noQ[:,0,0,:,x,3]+exp_noQ[:,0,0,:,x,4] + exp_noQ[:,0,0,:,x,5]+exp_noQ[:,0,0,:,x,6]) for x in range(len(cf.q))]

    mean_sw_RX = np.mean(sw_RX,axis = 1)/200_000
    mean_sw_low_eff_RX = np.mean(sw_low_eff_RX,axis = 1)/200_000
    mean_sw_noQ_RX = np.mean(sw_noQ_RX,axis = 1)/200_000

    mean_exp_RX = np.mean(exp_RX,axis = 1)/200_000
    mean_exp_low_eff_RX = np.mean(exp_low_eff_RX,axis = 1)/200_000
    mean_exp_noQ_RX = np.mean(exp_noQ_RX,axis = 1)/200_000

    fig, ax = plt.subplots(1,2,figsize = (14,5))

    a = [0,7,12,19]

    for x in a:
        ax[0].bar(a.index(x),       mean_exp_RX[2][x],          alpha = 0.8, color = colors[0], lw = 1.5, width = 0.2)
        ax[0].bar(a.index(x)+0.2,   mean_exp_noQ_RX[2][x],      alpha = 0.8, color = colors[1], lw = 1.5, width = 0.2)
        ax[0].bar(a.index(x)+0.4,   mean_exp_low_eff_RX[2][x],  alpha = 0.8, color = colors[2], lw = 1.5, width = 0.2)
        ax[1].bar(a.index(x),       mean_sw_RX[2][x],           alpha = 0.8, color = colors[0], lw = 1.5, width = 0.2)
        ax[1].bar(a.index(x)+0.2,   mean_sw_noQ_RX[2][x],       alpha = 0.8, color = colors[1], lw = 1.5, width = 0.2)
        ax[1].bar(a.index(x)+0.4,   mean_sw_low_eff_RX[2][x],   alpha = 0.8, color = colors[2], lw = 1.5, width = 0.2)

    positions = (0.1, 1.1, 2.1, 3.1)
    xlabels = ("0", "30", "50", "80")
    lines = [Line2D([0], [0], color = colors[x], alpha = 0.5, linewidth=4, linestyle='-') for x in [0,1,2]]
    ylabels = ['base ','noQ','lower eff']

    for i in [0,1]:
        ax[i].set_xticks(positions)
        ax[i].set_xticklabels(xlabels)
        ax[i].set_xlabel(r'app participation $a$ [%]',**hfont)
        ax[i].set_ylim(0,0.7)
        ax[i].set_ylabel(r'outbreak size $\Omega$',**hfont)
        ax[i].legend(lines, ylabels)
    #plt.savefig('Fig2',dpi = 300)
    plt.show()

def Fig3():
    omega_sw = np.array([(sw[:,0,1,:,x,2]+sw[:,0,1,:,x,3]+sw[:,0,1,:,x,4] + sw[:,0,1,:,x,5]+sw[:,0,1,:,x,6]) for x in range(len(cf.q))])
    DF_sw = (omega_sw)/np.array([(sw[:,0,1,:,x,4] + sw[:,0,1,:,x,5]) for x in range(len(cf.q))])

    mean_DF = np.mean(DF_sw,axis = 1)
    mean_omega = np.mean(omega_sw,axis = 1)/200_000

    fig, ax = plt.subplots(1,2,figsize = (14,5))

    for x in range(6):
        ax[0].plot(mean_omega[x],                            alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])
        ax[1].plot(((mean_omega[x]/mean_omega[x][0])-1)*100, alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])

    positions = (0, 6, 12, 18, 24)
    xlabels = ('0',"25","50", "75",'100')
    lines = [Line2D([0], [0], color = colors[x], marker = marker[x],linewidth=1.5, linestyle='-') for x in range(len(cf.q))]
    labels = ['q = 0','q = 0.1','q = 0.3','q = 0.5','q = 0.7','q = 0.9']

    for i in [0,1]:
        ax[i].set_xticks(positions)
        ax[i].set_xticklabels(xlabels)
        ax[i].set_xlabel(r'app participation $a$ [%]',**hfont)
    ax[0].legend(lines, labels)
    ax[0].set_ylim(0,0.75)
    ax[1].set_ylim(-75,5)
    ax[0].set_ylabel(r'outbreak size $\Omega$ ',**hfont)
    ax[1].set_ylabel(r'outbreak size reduction [%] ',**hfont)
    #plt.savefig('Fig3',dpi = 300)
    plt.show()

def Fig4():
    sw_low_eff_RX = [(sw[:,0,0,:,x,2] + sw[:,0,0,:,x,3]+sw[:,0,0,:,x,4] + sw[:,0,0,:,x,5]+sw[:,0,0,:,x,6]) for x in range(len(cf.q))]
    sw_noQ_RX =     [(sw_noQ[:,0,0,:,x,2] + sw_noQ[:,0,0,:,x,3]+sw_noQ[:,0,0,:,x,4] + sw_noQ[:,0,0,:,x,5]+sw_noQ[:,0,0,:,x,6]) for x in range(len(cf.q)) ]

    mean_sw_low_eff_RX = np.mean(sw_low_eff_RX,axis = 1)/200_000
    mean_sw_noQ_RX = np.mean(sw_noQ_RX,axis = 1)/200_000

    fig, ax = plt.subplots(2,2,figsize = (14,10))

    for x in range(6):
        ax[0,0].plot(mean_sw_noQ_RX[x],                                        alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])
        ax[0,1].plot(((mean_sw_noQ_RX[x]/mean_sw_noQ_RX[x][0])-1)*100,         alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])
        ax[1,0].plot(mean_sw_low_eff_RX[x],                                    alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])
        ax[1,1].plot(((mean_sw_low_eff_RX[x]/mean_sw_low_eff_RX[x][0])-1)*100, alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])

    positions = (0, 6, 12, 18, 24)
    xlabels = ('0',"25","50", "75",'100')

    for i in [0,1]:
        for j in [0,1]:
            ax[j,i].set_xticks(positions)
            ax[j,i].set_xticklabels(xlabels)
            ax[j,i].set_xlabel(r'app participation $a$ [%]',**hfont)
            ax[i,0].set_ylim(0,0.75)
            ax[i,1].set_ylim(-75,5)
            ax[i,0].set_ylabel(r'outbreak size $\Omega$ ',**hfont)
            ax[i,1].set_ylabel(r'outbreak size reduction [%] ',**hfont)

    lines = [Line2D([0], [0], color = colors[x], marker = marker[x],linewidth=1.5, linestyle='-') for x in range(len(cf.q))]
    labels = ['q = 0','q = 0.1','q = 0.3','q = 0.5','q = 0.7','q = 0.9']

    ax[1,1].legend(lines, labels)

    #plt.savefig('Fig4',dpi = 300)
    plt.show()

def Fig5():

    omega_lockdown = np.array([(lockdown[:,0,0,:,x,2] + lockdown[:,0,0,:,x,3]+lockdown[:,0,0,:,x,4] + lockdown[:,0,0,:,x,5]+lockdown[:,0,0,:,x,6]) for x in range(len(cf.q))])
    omega_no_lockdown = np.array([(no_lockdown[:,0,0,:,x,2] + no_lockdown[:,0,0,:,x,3]+no_lockdown[:,0,0,:,x,4] + no_lockdown[:,0,0,:,x,5]+no_lockdown[:,0,0,:,x,6]) for x in range(len(cf.q))])
    DF_lockdown = (omega_lockdown)/np.array([(lockdown[:,0,0,:,x,4] + lockdown[:,0,0,:,x,5]) for x in range(len(cf.q))])
    DF_no_lockdown = (omega_no_lockdown)/np.array([(no_lockdown[:,0,0,:,x,4] + no_lockdown[:,0,0,:,x,5]) for x in range(len(cf.q))])

    mean_DF_lockdown = np.mean(DF_lockdown,axis = 1)
    mean_omega_lockdown = np.mean(omega_lockdown,axis = 1)/200_000
    mean_DF_no_lockdown = np.mean(DF_no_lockdown,axis = 1)
    mean_omega_no_lockdown = np.mean(omega_no_lockdown,axis = 1)/200_000
    fig, ax = plt.subplots(2,2,figsize = (14,10))
    for x in range(6):
        ax[0,0].plot(mean_omega_no_lockdown[x],                                         alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])
        ax[0,1].plot(((mean_omega_no_lockdown[x]/mean_omega_no_lockdown[x][0])-1)*100,  alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])
        ax[1,0].plot(mean_omega_lockdown[x],                                            alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])
        ax[1,1].plot(((mean_omega_lockdown[x]/mean_omega_lockdown[x][0])-1)*100,        alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])

    positions = (0, 6, 12, 18, 24)
    xlabels = ('0',"25","50", "75",'100')

    for i in [0,1]:
        for j in [0,1]:
            ax[j,i].set_xticks(positions)
            ax[j,i].set_xticklabels(xlabels)
            ax[j,i].set_xlabel(r'app participation $a$ [%]',**hfont)
            ax[j,1].set_ylim(-85,5)
            ax[j,0].set_ylabel(r'outbreak size $\Omega$ ',**hfont)
            ax[j,1].set_ylabel(r'outbreak size reduction [%] ',**hfont)

    lines = [Line2D([0], [0], color = colors[x], marker = marker[x],linewidth=1.5, linestyle='-') for x in range(len(cf.q))]
    labels = ['q = 0','q = 0.1','q = 0.3','q = 0.5','q = 0.7','q = 0.9']
    ax[1,1].legend(lines, labels)
    #plt.savefig('Fig5',dpi = 300)
    plt.show()

def infectious():

    fig, ax = plt.subplots(1,3,figsize = (16,4))
    Iae = np.array([exp[:,0,1,:,x,1] for x in range(len(cf.q))])
    Ie = np.array([exp[:,0,1,:,x,0] for x in range(len(cf.q))])
    Ial = np.array([lockdown[:,0,0,:,x,1] for x in range(len(cf.q))])
    Il = np.array([lockdown[:,0,0,:,x,0] for x in range(len(cf.q))])
    Ianl = np.array([no_lockdown[:,0,0,:,x,1] for x in range(len(cf.q))])
    Inl = np.array([no_lockdown[:,0,0,:,x,0] for x in range(len(cf.q))])

    meanIe = np.mean(Ie, axis = 1)
    meanIae = np.mean(Iae, axis = 1)
    meanIl = np.mean(Il, axis = 1)
    meanIal = np.mean(Ial, axis = 1)
    meanInl = np.mean(Inl, axis = 1)
    meanIanl = np.mean(Ianl, axis = 1)

    for x in range(len(cf.q)):
        ax[0].plot(((meanIe[x]+meanIae[x])/(meanIe[x][0]+meanIae[x][0])-1)*100,     alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])
        ax[1].plot(((meanIl[x]+meanIal[x])/(meanIl[x][0]+meanIal[x][0])-1)*100,     alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])
        ax[2].plot(((meanInl[x]+meanIanl[x])/(meanInl[x][0]+meanIanl[x][0])-1)*100, alpha = 1, color = colors[x], lw = 1.5, marker = marker[x])

    positions = (0, 6, 12, 18, 24)
    xlabels = ('0',"0.25","0.5", "0.75",'1.0')
    for i in [0,1,2]:
        ax[i].set_xticks(positions)
        ax[i].set_xticklabels(xlabels)
        ax[i].set_xlabel(r'a',**hfont)

    lines = [Line2D([0], [0], color = colors[x], marker = marker[x],linewidth=1.5, linestyle='-') for x in range(len(cf.q))]
    labels = ['q = 0','q = 0.1','q = 0.3','q = 0.5','q = 0.7','q = 0.9']
    ax[0].legend(lines, labels)
    #plt.savefig('Fig6',dpi = 300)
    plt.show()
infectious()
