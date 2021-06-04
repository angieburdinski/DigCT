import numpy as np
#import bfmplot as bp
#from bfmplot import pl
import matplotlib.pyplot as pl
import pickle
from epipack.plottools import plot

import qsuite_config as cf
import matplotlib.ticker as mtick

with open('_qsuite/results_deleting_edges_20_new_N_meas_100/results_mean_std.p','rb') as f:
    data = pickle.load(f)

means = data['means']
stds = data['stds']

def sumres(result,compartments):
    return sum([result[C] for C in compartments])

def sumstd(result,compartments):
    return np.sqrt(sum([result[C]**2 for C in compartments]))

ia00 = 0
ia30 = 1
ia50 = 2
t = np.arange(len(means[0][0]['S']))

al = 0.3
fc = np.sqrt(cf.N_measurements)

plot_errors = False

for iph, phase in enumerate(cf.phases):

    #fig, ax = pl.subplots(1,1,figsize=(4,4))
    fig2, ax2 = pl.subplots(1,4,figsize=(15,4),sharex=True)
    ax = ax2[0]

    for ia in range(len(cf.a_s)):
        I = means[iph][ia]['Itot']
        StdI = stds[iph][ia]['Itot']/fc
        _p, = ax.plot(t,  sumres(means[iph][ia],['I_P','I_Pa','I_S','I_Sa','I_A','I_Aa']), label = f"a = {cf.a_s[ia]}")
        if plot_errors:
            ax.fill_between(t,  I-StdI, I+StdI,color=_p.get_color(),alpha=al,edgecolor='None')

    ax.set_xlim([0,t[-1]*0.55])
    ax.set_ylim([0,ax.get_ylim()[1]])
    ax.set_xlabel('time [days]')
    ax.set_ylabel('prevalence')
    ax.legend()


    Omeg0 = cf.N - means[iph][ia00]['Stot']
    StdOmeg0 = stds[iph][ia00]['Stot']/fc

    ax = ax2[1:]
    ax[0].plot(t,Omeg0,label='$\Omega(0)$',lw=3)

    for ia, a in enumerate(cf.a_s):
        if ia == 0:
            continue
        Omeg_a = cf.N - means[iph][ia]['Stot']
        StdOmeg_a = stds[iph][ia]['Stot']/fc
        dOm = Omeg0 - Omeg_a
        StdOm = sumstd({'0': StdOmeg0, '1': StdOmeg_a}, '01')

        relOm = 1 - Omeg_a/Omeg0
        StdRelOm = np.sqrt((StdOmeg_a/Omeg0)**2 + (StdOmeg0*Omeg_a/Omeg0**2)**2)

        ddOmdt = np.diff(dOm)

        #AuCorr = np.correlate(dOm, dOm, 'full')
        #AuCorr = AuCorr[len(AuCorr)//2:]
        #fig3, ax3 = pl.subplots()
        #ax3.plot(AuCorr)
        #pl.show()
        #print(AuCorr[:10])
        #AuCorr = np.correlate(dOm, dOm, 'valid')
        #print(np.mean(StdOm**2))
        #print(2*AuCorr[0])
        Cov = np.cov(dOm[1:], dOm[:-1])
        #print(Cov[0,1])
        StdddOmdt = np.sqrt(StdOm[1:]**2 + StdOm[:-1]**2 - 2*Cov[0,1])

        _p, = ax[0].plot(t,Omeg_a,label=f'$\Omega({a})$')
        if plot_errors:
            ax[0].fill_between(t,  Omeg_a-StdOmeg_a, Omeg_a+StdOmeg_a,color=_p.get_color(),alpha=al,edgecolor='None')

        _p, = ax[0].plot(t,dOm,'--',label=f'$\Omega(0) - \Omega({a})$')
        if plot_errors:
            ax[0].fill_between(t,  dOm-StdOm, dOm+StdOm,color=_p.get_color(),alpha=al,edgecolor='None')

        _p, = ax[1].plot(t,relOm,label=f'a={a}')
        if plot_errors:
            ax[1].fill_between(t,  relOm-StdRelOm, relOm+StdRelOm,color=_p.get_color(),alpha=al,edgecolor='None')

        _p, = ax[2].plot(t[:-1],ddOmdt,label=f'a={a}')
        if plot_errors:
            pass
            #ax[2].fill_between(t[:-1],  ddOmdt-StdddOmdt, ddOmdt+StdddOmdt,color=_p.get_color(),alpha=al,edgecolor='None')

        ax[0].set_xlabel('time [days]')
        ax[1].set_xlabel('time [days]')
        ax[2].set_xlabel('time [days]')
    ax[0].axvspan(20, 20, alpha=0.5, color='g')
    ax[0].set_ylabel('cumulative infections')
    ax[1].set_ylabel('relative averted infections (cumulative)')
    ax[2].set_ylabel('averted infections per day')

    ax[1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1,decimals=0))
    for col in range(4):
        ax2[col].set_xlim([0,t[-1]*0.55])
        #bp.strip_axis(ax2[col])
        ax2[col].legend()



    fig2.tight_layout()
    #fig2.savefig(f"figures/comparison_DigCT_free_and_lockdown_{phase.replace(' ','_')}.png",dpi=300)



pl.show()
