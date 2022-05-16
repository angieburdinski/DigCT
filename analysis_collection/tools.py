import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
import json
import numpy as np

plt.rcParams.update({'font.size': 10,
                    'font.sans-serif': 'Helvetica'})
c = [
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

folder  = 'tracing_sim/results_'
file = '/results_mean_err.npz'
a_positions = (0, 6, 12, 18, 24)
a_labels = ('0%', '25%', '50%', '75%', '100%')
a_positions_inset = (0, 12, 24)
a_labels_inset = ('0%', '50%', '100%')
N = 200_000

def data_main():
    liste = {"exp":"exp","exp_sw":"sw_exp","sw":"sw","random":"no_lockdown","lockdown":"lockdown"}
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

    data['exp']=           np.load(folder+'exponential_withQ_v2_NMEAS_100_ONLYSAVETIME_False'+file)
    data['exp_noQ'] =       np.load(folder+'exponential_withoutQ_NMEAS_100_ONLYSAVETIME_False'+file)
    data['exp_low_eff'] =   np.load(folder+'exponential_withQ_halfreact_NMEAS_100_ONLYSAVETIME_False'+file)
    data['lockdown'] =      np.load(folder+'smallworld_lockdown_withQ_v2_NMEAS_100_ONLYSAVETIME_False'+file)
    data['no_lockdown'] =   np.load(folder+'erdosrenyi_withQ_v2_NMEAS_100_ONLYSAVETIME_False'+file)
    data['sw'] =            np.load(folder+'smallworld_withQ_v3_NMEAS_100_ONLYSAVETIME_False'+file)
    data['sw_noQ'] =        np.load(folder+'smallworld_withoutQ_v2_NMEAS_100_ONLYSAVETIME_False'+file)
    data['sw_low_eff'] =    np.load(folder+'smallworld_withQ_halfreact_NMEAS_100_ONLYSAVETIME_False'+file)
    data['sw_exp'] =        np.load(folder+'smallworld_exponential_asc_withQ_NMEAS_100_ONLYSAVETIME_False'+file)
    data['sw_exp_']  =      np.load(folder+'smallworld_exponential_random_withQ_NMEAS_100_ONLYSAVETIME_False'+file)

    data['exp_noQ_y05'] =       np.load(folder+'exponential_withoutQ_y05'+file)
    data['exp_low_eff_y05'] =   np.load(folder+'exponential_withQ_halfreact_y05'+file)
    data['lockdown_y05'] =      np.load(folder+'smallworld_lockdown_withQ_y05'+file)
    data['no_lockdown_y05'] =   np.load(folder+'erdosrenyi_withQ_y05'+file)
    data['sw_y05'] =            np.load(folder+'smallworld_withQ_y05'+file)
    data['sw_noQ_y05'] =        np.load(folder+'smallworld_withoutQ_y05'+file)
    data['sw_low_eff_y05'] =    np.load(folder+'smallworld_withQ_halfreact_y05'+file)

    data['exp_y1'] =           np.load(folder+'exponential_withQ_y1_NMEAS_100_ONLYSAVETIME_False'+file)
    data['sw_exp_y1'] =        np.load(folder+'smallworld_exponential_asc_withQ_y1_NMEAS_100_ONLYSAVETIME_False'+file)
    data['exp_chi10'] =        np.load(folder+'exponential_withQ_chi10_NMEAS_100_ONLYSAVETIME_False'+file)
    data['sw_exp_chi10'] =     np.load(folder+'smallworld_exponential_asc_withQ_chi10_NMEAS_100_ONLYSAVETIME_False'+file)
    data['exp_z1'] =        np.load(folder+'exponential_withQ_z1_NMEAS_100_ONLYSAVETIME_False'+file)
    data['sw_exp_z1'] =     np.load(folder+'smallworld_exponential_asc_withQ_z1_NMEAS_100_ONLYSAVETIME_False'+file)

    keys = list(data.keys())
    for k in keys:
        data[k] = data[k]['mean']

    calc = {}
    for k in keys:
        calc[k] = {}
        O   = np.array([sum([data[k][:,x,0,i] for i in range(5)]) for x in range(4)])/N
        DF  = O/(np.array([sum([data[k][:,x,0,i] for i in [2,3]]) for x in range(4)])/N)
        red = [(((O[x]/O[x][0])-1)*100) for x in range(4)]
        calc[k]["absolute"] = {}
        calc[k]["reduction"] = {}
        for i in range(4):
            calc[k]["absolute"][str(np.round(DF[i][0]))] = list(O[i])
            calc[k]["reduction"][str(np.round(DF[i][0]))] = list(red[i])
        try:
            O   = np.array([sum([data[k][:,x,1,i] for i in range(5)]) for x in range(4)])/N
            DF  = O/(np.array([sum([data[k][:,x,1,i] for i in [2,3]]) for x in range(4)])/N)
            red = [(((O[x]/O[x][0])-1)*100) for x in range(4)]
            calc[k+'_y05'] = {}
            calc[k+'_y05']["absolute"] = {}
            calc[k+'_y05']["reduction"] = {}
            for i in range(4):
                calc[k+'_y05']["absolute"][str(np.round(DF[i][0]))] = list(O[i])
                calc[k+'_y05']["reduction"][str(np.round(DF[i][0]))] = list(red[i])
        except:
            pass
    with open('data_new.json', 'w') as outfile:
        json.dump(calc, outfile)

def FigS2():
    """
    This function generates Fig. S2
    """
    networks = ['sw','exp']

    fig, ax = plt.subplots(1,3,figsize = (18,4))

    R_0 = {}
    R_0['sw'] =  {}
    R_0['sw']['mean'] = np.load(folder+'smallworld_R0_y05_NMEAS_100_ONLYSAVETIME_False'+file)
    R_0['sw']['all'] = np.load(folder+'smallworld_R0_y05_NMEAS_100_ONLYSAVETIME_False/results.npy')

    R_0['exp'] =  {}
    R_0['exp']['mean'] = np.load(folder+'exponential_R0_y05_NMEAS_100_ONLYSAVETIME_False'+file)
    R_0['exp']['all'] = np.load(folder+'exponential_R0_y05_NMEAS_100_ONLYSAVETIME_False/results.npy')

    for k in networks:
        R_0[k]['mean'] = R_0[k]['mean']['mean']

    _R0 = np.linspace(0.1,10,51)
    lines = [Line2D([0], [0], color = c[i], alpha = 1, linewidth=2, linestyle='-') for i in range(2)]
    labels = ['0% app user', '30% app user']

    for k_ix, k in enumerate(networks):
        axin = ax[k_ix].inset_axes([0.54, 0.1,0.4, 0.4])

        for a_ix in range(2):
            mean = np.array(sum([R_0[k]['mean'][a_ix,0,:,i] for i in range(5)]))/N
            var = np.var(np.array(sum([R_0[k]['all'][:,a_ix,0,:,i] for i in range(5)]))/N, axis = 0)/mean**2
            ax[k_ix].plot(mean, color = c[a_ix])
            axin.plot(var, color = c[a_ix])

        axin.set_xticks((0,10,20,30,40,50))
        axin.set_xticklabels((int(_R0[0]),int(_R0[10]),int(_R0[20]),int(_R0[30]),int(_R0[40]),int(_R0[50])))
        axin.set_ylabel(r'CV of $\langle\Omega\rangle/N$')

        ax[k_ix].set_ylabel(r'$\langle\Omega\rangle/N$')
        ax[k_ix].set_ylim(0,1)
        ax[k_ix].legend(lines,labels)
        ax[k_ix].set_xlabel('$R_0$')
        ax[k_ix].set_xticks((0,10,20,30,40,50))
        ax[k_ix].set_xticklabels((int(_R0[0]),int(_R0[10]),int(_R0[20]),int(_R0[30]),int(_R0[40]),int(_R0[50])))


    DF = {}
    DF['sw'] =  np.load(folder+'smallworld_DF_y05_NMEAS_100_ONLYSAVETIME_False'+file)
    DF['exp'] = np.load(folder+'exponential_DF_y05_NMEAS_100_ONLYSAVETIME_False'+file)
    for k in networks:
        DF[k] = DF[k]['mean']
        DF[k] = (np.array([sum([DF[k][:,x,i] for i in range(5)]) for x in range(6)])/N)/\
                      (np.array([sum([DF[k][:,x,i] for i in [2,3]]) for x in range(6)])/N)
    for i in range(6):
        ax[2].plot(DF["exp"][i],color = c[i],linewidth=2, ls = '-')
        ax[2].plot(DF["sw"][i],color = c[i],linewidth=2,ls = ':')


    lines_DF = [Line2D([0], [0], color = c[i], alpha = 1, linewidth=2, linestyle='-') for i in [1,2,3,4,5]]
    ylabels_DF = ['q = 0.1','q = 0.3','q = 0.5','q = 0.7','q = 0.9']

    ax[2].legend(lines_DF,ylabels_DF)
    ax[2].set_xticks(a_positions)
    ax[2].set_xticklabels(a_labels)
    ax[2].set_ylabel('dark factor DF')
    ax[2].set_xlabel('app participation $a$ [%]')

    plt.show()

def standard_plot(sims):

    """ This function generates for a string-list of simulation names a standard plot"""

    with open('data_new.json') as json_file:
        data = json.load(json_file)

    a = np.linspace(0,24,25)
    lb = "2.0"
    ub = "12.0"
    mid = "4.0"
    if len(sims) > 2:
        fig, ax = plt.subplots(2,int(len(sims)/2),figsize = (10,4), sharey = True)
    else:
        fig, ax = plt.subplots(1,2,figsize = (25,10), sharey = 'row')

    axs = ax.flatten()

    for ix, res in enumerate(sims):
        axin = axs[ix].inset_axes([0.19, 0.13,0.35, 0.3])
        for i in [ub,lb]:
            axs[ix].plot(data[res]['reduction'][i], color = "k",  alpha = 1)
            axin.plot(data[res]['absolute'][i],     color = "k",  alpha = 1)
        axs[ix].plot(data[res]['reduction'][mid],   color = "k",  alpha = 1, marker = ".")
        axin.plot(data[res]['absolute'][mid],       color = "k",  alpha = 1, marker = ".")
        axs[ix].fill_between(a,data[res]['reduction'][ub],data[res]['reduction'][lb],   color = "grey", alpha = 0.3)
        axin.fill_between(a,data[res]['absolute'][ub],data[res]['absolute'][lb],        color = "grey", alpha = 0.3)
        axs[ix].set_xticks(a_positions)
        axs[ix].set_xticklabels(a_labels )

        if data[sims[int(len(sims)/2)]]['reduction'][lb][-1] <= np.min(data[sims[0]]['reduction'][lb][-1]):
            axs[ix].set_ylim(np.min(data[sims[int(len(sims)/2)]]['reduction'][lb])-5,0)
        else:
            axs[ix].set_ylim(np.min(data[sims[0]]['reduction'][lb])-5,0)
        if ix < int(len(sims)/2):

            for i in [ub,lb]:
                axs[ix].fill_between(a,data[sims[0]]['reduction'][ub],data[sims[0]]['reduction'][lb],   color = "navy", alpha = 0.1)
                axin.fill_between(a,data[sims[0]]['absolute'][ub],data[sims[0]]['absolute'][lb],        color = "navy", alpha = 0.1)



            axin.set_ylim(np.min(data[sims[0]]['absolute'][lb])-0.05,np.max(data[sims[0]]['absolute'][ub])+0.05)
            if len(sims) <= 2:
                axs[ix].set_xlabel(r'app participation $a$' )
        else:

            for i in [ub,lb]:
                axs[ix].fill_between(a,data[sims[int(len(sims)/2)]]['reduction'][ub],data[sims[int(len(sims)/2)]]['reduction'][lb],   color = "navy", alpha = 0.1)
                axin.fill_between(a,data[sims[int(len(sims)/2)]]['absolute'][ub],data[sims[int(len(sims)/2)]]['absolute'][lb],        color = "navy", alpha = 0.1)


            axin.set_ylim(np.min(data[sims[int(len(sims)/2)]]['absolute'][lb])-0.05,np.max(data[sims[int(len(sims)/2)]]['absolute'][ub])+0.05)
            axs[ix].set_xlabel(r'app participation $a$' )


        axs[ix].yaxis.set_major_formatter(mtick.PercentFormatter())
        axin.set_xticks(a_positions_inset)
        axin.set_xticklabels(a_labels_inset )
        axin.text(0,1,r'$\langle\Omega\rangle$/N',transform=axin.transAxes,ha='right',va='bottom' )
        #axs[ix].set_title(res)
    axs[0].set_ylabel(r'outbreak size reduction' )
    if len(sims) > 2:
        axs[int(len(sims)/2)].set_ylabel(r'outbreak size reduction' )
    plt.show()

def FigS3():
    """ This function generates Fig. S3 """
    sims = ["sw_y05","sw_low_eff_y05","sw_noQ_y05","exp_y05","exp_low_eff_y05","exp_noQ_y05"]
    standard_plot(sims)

def FigS4():
    """ This function generates Fig. S4 """
    sims = ['exp_y05','exp_y1', 'exp_chi10','exp_z1', 'sw_exp_y05','sw_exp_y1','sw_exp_chi10','sw_exp_z1']
    standard_plot(sims)

def FigS6():
    """ This function generates Fig. S6 """
    sims = ["sw_exp_y05","sw_exp__y05"]
    standard_plot(sims)

if __name__ == "__main__":

    load_export_data()
    data_main()
    #FigS2()
    FigS3()
    #FigS4()
    #FigS6()
