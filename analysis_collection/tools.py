import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
import json
import numpy as np
from pathlib import Path

plt.rcParams.update({'font.size': 8,
                    'font.sans-serif': 'Helvetica'})
c1 = ['k','grey']
"ER","WS","EXP","WS-EXP_asc"
c = {
    "ER":'#333333',
    "WS":'#5540BF',
    "EXP":'#FE6100',
    "WS-EXP_asc":'#FFB000',
    "WS-EXP":'#FFB000',
    "WS-EXP_rnd":'#FFB000'
 }


folder  = 'tracing_sim/results_'
file = '/results_mean_err.npz'
a_positions = (0, 6, 12, 18, 24)
a_labels = ('0%', '25%', '50%', '75%', '100%')
a_positions_inset = (0, 12, 24)
a_labels_inset = ('0%', '50%', '100%')
N = 200_000
a = np.linspace(0,24,25)

def data_main():
    liste = {"exp":"EXP","exp_sw":"WS-EXP_asc","sw":"WS","random":"ER","lockdown":"lockdown"}
    data = {}
    data["a"] = list(np.linspace(0,1,25))
    with open('data_new.json') as json_file:
        data_new = json.load(json_file)
    for k,v in liste.items():
        data[k] = data_new[v+"_y05"]["absolute"]
    with open('data_main.json', 'w') as outfile:
        json.dump(data, outfile)
def load_export_data():
    """
    This function loads the simulation data and generates a file in which all
    computed results are saved.
    Return
    -------

    Dictionary with absolute outbreak size, rounded initial dark factor (DF_0) and relative outbreak size reduction
    for different settings.
    """
    data = {}

    data['ER'] =      np.load(folder+'erdosrenyi_withQ_v2_NMEAS_100_ONLYSAVETIME_False'+file)
    data['ER_y05'] =   np.load(folder+'erdosrenyi_withQ_y05'+file)
    data['ER_noQ_y05'] =       np.load(folder+'erdosrenyi_withoutQ_y05'+file)
    data['ER_low_eff_y05'] =   np.load(folder+'erdosrenyi_withQ_halfreact_y05'+file)
    data['ER_y1'] =       np.load(folder+'erdosrenyi_withQ_y1'+file)
    data['ER_z1'] =   np.load(folder+'erdosrenyi_withQ_y05_z1'+file)
    data['ER_chi10'] =   np.load(folder+'erdosrenyi_withQ_y05_chi10'+file)

    data['EXP'] =           np.load(folder+'exponential_withQ_v2_NMEAS_100_ONLYSAVETIME_False'+file)
    data['EXP_noQ'] =       np.load(folder+'exponential_withoutQ_NMEAS_100_ONLYSAVETIME_False'+file)
    data['EXP_low_eff'] =   np.load(folder+'exponential_withQ_halfreact_NMEAS_100_ONLYSAVETIME_False'+file)
    data['EXP_noQ_y05'] =       np.load(folder+'exponential_withoutQ_y05'+file)
    data['EXP_low_eff_y05'] =   np.load(folder+'exponential_withQ_halfreact_y05'+file)
    data['EXP_y1'] =           np.load(folder+'exponential_withQ_y1_NMEAS_100_ONLYSAVETIME_False'+file)
    data['EXP_z1'] =        np.load(folder+'exponential_withQ_z1_NMEAS_100_ONLYSAVETIME_False'+file)
    data['EXP_chi10'] =        np.load(folder+'exponential_withQ_chi10_NMEAS_100_ONLYSAVETIME_False'+file)

    data['WS'] =            np.load(folder+'smallworld_withQ_v3_NMEAS_100_ONLYSAVETIME_False'+file)
    data['WS_noQ'] =        np.load(folder+'smallworld_withoutQ_v2_NMEAS_100_ONLYSAVETIME_False'+file)
    data['WS_low_eff'] =    np.load(folder+'smallworld_withQ_halfreact_NMEAS_100_ONLYSAVETIME_False'+file)
    data['WS_y05'] =            np.load(folder+'smallworld_withQ_y05'+file)
    data['WS_noQ_y05'] =        np.load(folder+'smallworld_withoutQ_y05'+file)
    data['WS_low_eff_y05'] =    np.load(folder+'smallworld_withQ_halfreact_y05'+file)
    data['WS_y1'] =           np.load(folder+'smallworld_withQ_y1'+file)
    data['WS_z1'] =        np.load(folder+'smallworld_withQ_z1'+file)
    data['WS_chi10'] =        np.load(folder+'smallworld_withQ_chi10'+file)

    data['WS-EXP_asc'] =        np.load(folder+'smallworld_exponential_asc_withQ_NMEAS_100_ONLYSAVETIME_False'+file)
    data['WS-EXP_asc_noQ_y05'] =       np.load(folder+'smallworld_exponential_withoutQ_y05'+file)

    data['WS-EXP_asc_low_eff_y05'] =        np.load(folder+'smallworld_exponential_withQ_halfreact_y05'+file)
    data['WS-EXP_rnd']  =      np.load(folder+'smallworld_exponential_random_withQ_NMEAS_100_ONLYSAVETIME_False'+file)
    data['WS-EXP_asc_y1'] =        np.load(folder+'smallworld_exponential_asc_withQ_y1_NMEAS_100_ONLYSAVETIME_False'+file)
    data['WS-EXP_asc_chi10'] =     np.load(folder+'smallworld_exponential_asc_withQ_chi10_NMEAS_100_ONLYSAVETIME_False'+file)
    data['WS-EXP_asc_z1'] =     np.load(folder+'smallworld_exponential_asc_withQ_z1_NMEAS_100_ONLYSAVETIME_False'+file)

    data['lockdown'] =      np.load(folder+'smallworld_lockdown_withQ_v2_NMEAS_100_ONLYSAVETIME_False'+file)
    data['lockdown_y05'] =      np.load(folder+'smallworld_lockdown_withQ_y05'+file)

    keys = list(data.keys())
    for k in keys:
        data[k] = data[k]['mean']

    calc = {}
    for k in keys:
        calc[k] = {}
        calc[k]["absolute"] = {}
        calc[k]["reduction"] = {}
        x_len = np.size(data[k],1)

        O   = np.array([sum([data[k][:,x,0,i] for i in range(5)]) for x in range(x_len)])/N
        DF  = O/(np.array([sum([data[k][:,x,0,i] for i in [2,3]]) for x in range(x_len)])/N)
        red = [(((O[x]/O[x][0])-1)*100) for x in range(x_len)]
        for i in range(x_len):
            calc[k]["absolute"][str(np.round(DF[i][0]))] = list(O[i])
            calc[k]["reduction"][str(np.round(DF[i][0]))] = list(red[i])

        try:
            O   = np.array([sum([data[k][:,x,1,i] for i in range(5)]) for x in range(x_len)])/N
            DF  = O/(np.array([sum([data[k][:,x,1,i] for i in [2,3]]) for x in range(x_len)])/N)
            red = [(((O[x]/O[x][0])-1)*100) for x in range(x_len)]
            calc[k+'_y05'] = {}
            calc[k+'_y05']["absolute"] = {}
            calc[k+'_y05']["reduction"] = {}
            for i in range(x_len):
                calc[k+'_y05']["absolute"][str(np.round(DF[i][0]))] = list(O[i])
                calc[k+'_y05']["reduction"][str(np.round(DF[i][0]))] = list(red[i])
        except:
            pass
    with open('data_new.json', 'w') as outfile:
        json.dump(calc, outfile)
def Fig_SI_criticality_UA():
    """
    This function generates Fig. S2
    """
    lss = ['solid', 'dashed', 'dotted','dashdot']
    folder  = 'tracing_sim/results_'
    file = '/results_mean_err.npz'
    networks = ['ER','WS','EXP','WS-EXP_asc']

    fig, ax = plt.subplots(1,5,figsize = (18,4))

    R_0 = {}
    for i_n, n in enumerate(networks):
        R_0[n] = {}
        R_0[n]['mean'] = np.load(folder+'R0_analysis_y05_NMEAS_100_ONLYSAVETIME_False'+file)['mean'][:,:,i_n]
        R_0[n]['all'] = np.load(folder+'R0_analysis_y05_NMEAS_100_ONLYSAVETIME_False/results.npy')[:,:,:,i_n]

    _R0 = np.linspace(0,10,51)

    lines = [Line2D([0], [0], color = 'k', linestyle=lss[i*2]) for i in range(2)]

    labels = ['0% app user', '30% app user']

    for i_n, n in enumerate(networks):
        axin = ax[i_n].inset_axes([0.54, 0.3,0.4, 0.25])
        for a_ix in range(2):
            mean = R_0[n]['mean'][a_ix,:]/N
            var = np.var(R_0[n]['all'][:,a_ix,:]/N, axis = 0)/mean**2
            ax[i_n].plot(mean, color = c[n], ls = lss[a_ix*2])
            axin.plot(var, color = c[n], ls = lss[a_ix*2])

        axin.set_xticks((0,10,20,30,40,50))
        axin.set_xticklabels((int(_R0[0]),int(_R0[10]),int(_R0[20]),int(_R0[30]),int(_R0[40]),int(_R0[50])))
        axin.set_title(r'CV of $\langle\Omega\rangle/N$')#,loc = 'top')

        ax[i_n].set_ylabel(r'outbreak size $\langle\Omega\rangle/N$', loc = 'bottom')
        ax[i_n].set_ylim(0,1)
        ax[i_n].legend(lines,labels)
        ax[i_n].set_xlabel('basic reproductive ratio $R_0$',loc = 'right')
        ax[i_n].set_xticks((0,10,20,30,40,50))
        ax[i_n].set_xticklabels((int(_R0[0]),int(_R0[10]),int(_R0[20]),int(_R0[30]),int(_R0[40]),int(_R0[50])))


    DF = {}

    data = {}
    data['ER'] =         np.load(folder+'erdosrenyi_withQ_y05'+file)['mean'][:,:,0,:]
    data['EXP'] =        np.load(folder+'exponential_withQ_v2_NMEAS_100_ONLYSAVETIME_False'+file)['mean'][:,:,1,:]
    data['WS'] =         np.load(folder+'smallworld_withQ_y05'+file)['mean'][:,:,0,:]
    data['WS-EXP_asc'] = np.load(folder+'smallworld_exponential_asc_withQ_NMEAS_100_ONLYSAVETIME_False'+file)['mean'][:,:,1,:]
    for n in networks:
        DF[n] = (np.array([sum([data[n][:,x,i] for i in range(5)]) for x in range(4)])/N)/\
                (np.array([sum([data[n][:,x,i] for i in [2,3]]) for x in range(4)])/N)
        for i in range(1,4):
            ax[4].plot(DF[n][i], color = c[n], ls = lss[i-1])

    lines_DF = [Line2D([0], [0], color = 'k', linestyle=lss[i]) for i in range(3)]
    ylabels_DF = ['q = 0.1','q = 0.3','q = 0.5']

    ax[4].legend(lines_DF,ylabels_DF)
    ax[4].set_xticks(a_positions)
    ax[4].set_xticklabels(a_labels)

    ax[4].set_ylabel('under-ascertainment $UA$', loc = 'bottom')
    ax[4].text(1,-0.1,'app participation $a$',transform=ax[4].transAxes,va='bottom',ha='right')
    for i in range(5):
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
    plt.show()
def standard_plot(senses, netw):

    """ This function generates for a string-list of simulation names a standard plot"""

    with open('data_new.json') as json_file:
        data = json.load(json_file)

    a = np.linspace(0,24,25)
    lb = "2.0"
    ub = "12.0"
    mid = "4.0"

    if len(netw) == 1 and len(senses) != 1:
        fig, axs = plt.subplots(2,len(senses),figsize = (10,4), sharey = True)
    elif len(netw) != 1 and len(senses) == 1:
        fig, axs = plt.subplots(len(netw),2,figsize = (10,4), sharey = True)
    elif len(netw) == 1 and len(senses) == 1:
        fig, axs = plt.subplots(2,2,figsize = (10,4), sharey = True)
    else:
        fig, axs = plt.subplots(len(netw),len(senses),figsize = (20,8), sharey = True)

    for i_x, sim in enumerate(senses):
        for i_n, net in enumerate(netw):
            axin = axs[i_n,i_x].inset_axes([0.17, 0.13,0.35, 0.3])
            axs[i_n,i_x].spines['bottom'].set_visible(False)
            axs[i_n,i_x].spines['right'].set_visible(False)
            axin.spines['top'].set_visible(False)
            axin.spines['right'].set_visible(False)

            axs[i_n,i_x].xaxis.tick_top()
            if i_n == 0:
                axin.set_yticks([0.5,0.8])
            if i_n == 1 or i_n ==3 :
                axin.set_yticks([0.2,0.6])


            for i in [ub,lb]:
                axs[i_n,i_x].fill_between(a,data[net+senses[0]]['reduction'][ub],data[net+senses[0]]['reduction'][lb],  color = c[net], alpha = 0.2)
                axin.fill_between(a,data[net+senses[0]]['absolute'][ub],data[net+senses[0]]['absolute'][lb],      color = c[net], alpha = 0.2)

            axin.set_ylim(np.min(data[net+sim]['absolute'][lb])-0.05,np.max(data[net+sim]['absolute'][ub])+0.05)
            #axs[0,i_x].text(1.1,1.15,'app participation $a$ ',transform=axs[0,i_x].transAxes,va='top',ha='right')

            for i in [ub,lb]:
                axs[i_n,i_x].plot(data[net+sim]['reduction'][i], color = "k")
                axin.plot(data[net+sim]['absolute'][i], color = "k")

            axs[i_n,i_x].plot(data[net+sim]['reduction'][mid], color = "k", marker = "o",mec='w')
            axin.plot(data[net+sim]['absolute'][mid], color = "k",  marker = ".",mec='w')

            axs[i_n,i_x].fill_between(a,data[net+sim]['reduction'][ub],data[net+sim]['reduction'][lb], color ='#999999', alpha = 0.2)
            axin.fill_between(a,data[net+sim]['absolute'][ub],data[net+sim]['absolute'][lb], color = '#999999', alpha = 0.2)
            axs[i_n,i_x].set_xticks(a_positions)
            axs[i_n,i_x].set_xticklabels(a_labels)
            axs[i_n,0].set_ylabel(r'outbreak size reduction', loc = 'bottom')
            axs[i_n,i_x].yaxis.set_major_formatter(mtick.PercentFormatter())
            axin.set_xticks(a_positions_inset)
            axin.set_xticklabels(a_labels_inset )
            axin.text(0,1,r'$\langle\Omega\rangle$/N',transform=axin.transAxes,ha='right',va='bottom' )
            axin.text(1.1,-0.1,'$a$ ',transform=axin.transAxes)
    fig.tight_layout()
    plt.show()
def Fig_sensitivity_all():
    senses = ['_y05',"_low_eff_y05", '_noQ_y05','_y1',"_z1","_chi10"]
    netw = ["ER","WS","EXP","WS-EXP_asc"]
    standard_plot(senses,netw)
def Fig_SI_ascrand():
    """ This function generates ascrand """
    senses = ['_y05']
    netw = ["WS-EXP_asc","WS-EXP_rnd"]

    standard_plot(senses,netw)
def Fig_4panel_lockdown():

    p = str(Path(__file__).parents[0]) + '/tracing_sim/results_panels4_lockdown'
    networks = ['ER','WS','EXP','WS-EXP']
    lockdown = [True,False]
    data = np.concatenate((np.load(str(p)+'_ER_SW_200k/results_mean_err.npz')['mean'], np.load(str(p)+'_ER_SW_exp_200k/results_mean_err.npz')['mean']))
    print(data.shape)
    #data_ER_0_2 = np.load(str(p)+'_ER_200k_04/results_mean_err.npz')['mean']
    #data_ER_0_4 = np.load(str(p)+'_ER_200k/results_mean_err.npz')['mean']

    calc = {}
    for i_n,n in enumerate(networks):
        calc[n]= {}
        for i_l,l in enumerate(lockdown):
            calc[n][str(l)]= {}
            for i_q, q in enumerate([0.1,0.3,0.5]):
                O = np.sum(data[i_n,i_q,:,i_l], axis = 1)/N
                CX = sum([data[i_n,i_q,0,i_l][i] for i in [2,3]])/N
                DF = np.round(O/CX)
                DF = str(DF[0])
                calc[n][str(l)][DF] = {}
                calc[n][str(l)][DF]['absolute'] = O
                calc[n][str(l)][DF]['reduction'] = (O/O[0]-1)*100

    fig, ax = plt.subplots(4,2, sharey = True)

    lb = "2.0"
    ub = "12.0"
    mid = "4.0"

    for i_n,n in enumerate(networks):
        for i_l,l in enumerate(reversed(lockdown)):
            l = str(l)
            axin = ax[i_n,i_l].inset_axes([0.15, 0.15,0.3, 0.3])
            if i_n== 0:
                axin.set_ylim([0,1])
            else:
                axin.set_ylim([0,0.7])
            """
            if i_n== 0:
                axin.set_ylim([0,1])

            if i_n== 2:
                axin.set_ylim([0,0.7])
            if i_n== 3:
                axin.set_ylim([0,0.7])
            if i_n== 1 and i_l == 0:
                axin.set_ylim([0,0.7])
            if i_n== 1 and i_l == 1:
                axin.set_ylim([0,0.1])
            """

            for i in [ub,lb]:
                ax[i_n,i_l].plot(calc[n][l][i]['reduction'], color = "k",  alpha = 1)
                axin.plot(calc[n][l][i]['absolute'], color = "k",  alpha = 1, lw = 0.4)

            ax[i_n,i_l].plot(calc[n][l][mid]['reduction'], color = "k", alpha = 1, marker = "o",mec='w')
            axin.plot(calc[n][l][mid]['absolute'], color = "k", alpha = 1,marker = ".",markersize = 0.5,markeredgewidth = 0.1,mec='w')
            ax[i_n,i_l].fill_between(a,calc[n][l][ub]['reduction'],calc[n][l][lb]['reduction'], color = c[n], alpha = 0.3)
            axin.fill_between(a,calc[n][l][ub]['absolute'],calc[n][l][lb]['absolute'], color = c[n], alpha = 0.3)


            ax[i_n,i_l].xaxis.tick_top()
            ax[i_n,i_l].set_xticks(a_positions)
            ax[i_n,i_l].set_xticklabels(a_labels)
            axin.set_xticks(a_positions_inset)
            axin.set_xticklabels(a_labels_inset)

            ax[i_n,0].set_ylabel('outbreak size reduction',loc='bottom')

            axin.text(0.1,1.1,r'$\langle\Omega\rangle$/N',transform=axin.transAxes,ha='right',va='bottom' )
            ax[i_n,i_l].yaxis.set_major_formatter(mtick.PercentFormatter())

            ax[i_n,i_l].spines['bottom'].set_visible(False)
            ax[i_n,i_l].spines['right'].set_visible(False)
            axin.spines['top'].set_visible(False)
            axin.spines['right'].set_visible(False)
    plt.show()
def Fig_periodic_lockdown():

    import pickle

    c = ['k','grey','darkgreen']
    times = [0,30,34,40]
    data_periodic_lockdown = {}

    for i_t, t in enumerate(times):
        if t == 0:
            continue
        with open('tracing_sim/results_deleting_edges_'+str(t)+'_y05_N_meas_100/results_mean_std.p','rb') as f:
            data = pickle.load(f)
        data_periodic_lockdown[t] = data['means']

    def sumres(result,compartments):
        return sum([result[C] for C in compartments])

    ia00 = 0
    ia30 = 1
    ia50 = 2
    a_s = [0,0.3,0.5]

    fig, axs = plt.subplots(len(times),4,figsize = (13,13),sharex=True, sharey = 'col')

    for i_t, t in enumerate(times):
        if t == 0:
            sim_time = np.arange(len(data_periodic_lockdown[times[1]][t][0]['S']))
        else:
            sim_time = np.arange(len(data_periodic_lockdown[t][1][0]['S']))
            x = [t,t,t,1e100]

        for ia, a in enumerate(a_s):
            if t == 0:
                I = data_periodic_lockdown[times[1]][0][ia]['Itot']
                Omeg_a = N - data_periodic_lockdown[times[1]][0][ia]['Stot']
                Omeg0 = N - data_periodic_lockdown[times[1]][0][ia00]['Stot']
            else:
                I = data_periodic_lockdown[t][1][ia]['Itot']
                Omeg0 = N - data_periodic_lockdown[t][1][ia00]['Stot']
                Omeg_a = N - data_periodic_lockdown[t][1][ia]['Stot']


            if ia == 0:
                axs[i_t,0].plot(sim_time,  I, label = f"a = {a}",color = c[ia])
                axs[i_t,1].plot(sim_time,Omeg0,label='$\Omega(0)$',color = c[ia])
                continue

            dOm = Omeg0 - Omeg_a
            relOm = 1 - Omeg_a/Omeg0
            ddOmdt = np.diff(dOm)

            axs[i_t,0].plot(sim_time,  I, label = f"a = {a}",color = c[ia])
            _p = axs[i_t,1].plot(sim_time,Omeg_a,label=f'$\Omega({a})$',color = c[ia])
            axs[i_t,1].plot(sim_time,dOm,'--',label=f'$\Omega(0) - \Omega({a})$', color = _p[0].get_color())
            axs[i_t,2].plot(sim_time,relOm,label=f'a={a}',color = c[ia])
            axs[i_t,3].plot(sim_time[:-1],ddOmdt,label=f'a={a}',color = c[ia])

        axs[i_t,2].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1,decimals=0))

        for i in range(4):
            if t!= 0:
                axs[i_t,i].axvspan(x[0], x[0]*2, alpha=0.3, color='grey')
                axs[i_t,i].axvspan(x[0]*3, max(sim_time), alpha=0.3, color='grey')
            axs[3,i].set_xlabel('time [days]')
            axs[0,i].legend()
            axs[i_t,i].set_xlim([0,180])

        axs[i_t,0].set_ylabel('prevalence')
        axs[i_t,1].set_ylabel('cumulative infections')
        axs[i_t,2].set_ylabel('relative averted infections \n (cumulative)')
        axs[i_t,3].set_ylabel('averted infections per day')
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_main()
    load_export_data()
    Fig_SI_criticality_UA()
    Fig_sensitivity_all()
    Fig_SI_ascrand()
    Fig_4panel_lockdown()
    Fig_periodic_lockdown()
