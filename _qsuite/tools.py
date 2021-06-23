import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
hfont = {'fontname':'Helvetica'}
plt.rcParams.update({'font.size': 8})

import pickle
import gzip
import json
import numpy as np
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
def load_export_data():
    """
    This function loads the simulation data and generates a file in which all
    computed results are saved.
    Return
    -------
    Dictionary with absolute outbreak size, rounded initial dark factor (DF_0) and relative outbreak size reduction
    for different settings.
    """
    exp =           np.load('_qsuite/tracing_sim/results_exponential_withQ_v2_NMEAS_100_ONLYSAVETIME_False/results_mean_err.npz')
    exp_noQ =       np.load('_qsuite/tracing_sim/results_exponential_withoutQ_NMEAS_100_ONLYSAVETIME_False/results_mean_err.npz')
    exp_low_eff =   np.load('_qsuite/tracing_sim/results_exponential_withQ_halfreact_NMEAS_100_ONLYSAVETIME_False/results_mean_err.npz')
    lockdown =      np.load('_qsuite/tracing_sim/results_smallworld_lockdown_withQ_v2_NMEAS_100_ONLYSAVETIME_False/results_mean_err.npz')
    no_lockdown =   np.load('_qsuite/tracing_sim/results_erdosrenyi_withQ_v2_NMEAS_100_ONLYSAVETIME_False/results_mean_err.npz')
    sw =            np.load('_qsuite/tracing_sim/results_smallworld_withQ_v3_NMEAS_100_ONLYSAVETIME_False/results_mean_err.npz')
    sw_noQ =        np.load('_qsuite/tracing_sim/results_smallworld_withoutQ_v2_NMEAS_100_ONLYSAVETIME_False/results_mean_err.npz')
    sw_low_eff =    np.load('_qsuite/tracing_sim/results_smallworld_withQ_halfreact_NMEAS_100_ONLYSAVETIME_False/results_mean_err.npz')
    sw_exp =        np.load('_qsuite/tracing_sim/results_smallworld_exponential_asc_withQ_NMEAS_100_ONLYSAVETIME_False/results_mean_err.npz')
    sw_exp_  =      np.load('_qsuite/tracing_sim/results_smallworld_exponential_random_withQ_NMEAS_100_ONLYSAVETIME_False/results_mean_err.npz')

    data = {}
    data["exp"] =           exp['mean']
    data["exp_noQ"] =       exp_noQ['mean']
    data["exp_low_eff"] =   exp_low_eff['mean']
    data["lockdown"] =      lockdown['mean']
    data["no_lockdown"] =   no_lockdown['mean']
    data["sw"] =            sw['mean']
    data["sw_noQ"] =        sw_noQ['mean']
    data["sw_low_eff"] =    sw_low_eff['mean']
    data["sw_exp"] =        sw_exp['mean']
    data["sw_exp_"] =       sw_exp_['mean']


    datalist = [exp,exp_noQ,exp_low_eff,lockdown,no_lockdown,sw,sw_noQ,sw_low_eff,sw_exp,sw_exp_]
    stringlist = ["exp","exp_noQ","exp_low_eff","lockdown","no_lockdown","sw","sw_noQ","sw_low_eff","sw_exp","sw_exp_"]

    data_dict = {}

    for k,v in data.items():

        data_dict[k] = {}
        data_dict[k]["O"] = np.array([sum([data[k][:,x,0,i] for i in range(5)]) for x in range(4)])/200_000
        data_dict[k]["DF"] = (data_dict[k]["O"])/(np.array([sum([data[k][:,x,0,i] for i in [2,3]]) for x in range(4)])/200_000)
        data_dict[k]["red"] =  [(((data_dict[k]["O"][x]/data_dict[k]["O"][x][0])-1)*100) for x in range(4)]
        try:
            data_dict[k]["O_y0.5"] = np.array([sum([data[k][:,x,1,i] for i in range(5)]) for x in range(4)])/200_000
            data_dict[k]["DF_y0.5"] = (data_dict[k]["O_y0.5"])/(np.array([sum([data[k][:,x,1,i] for i in [2,3]]) for x in range(4)])/200_000)
            data_dict[k]["red_y0.5"] =   [(((data_dict[k]["O_y0.5"][x]/data_dict[k]["O_y0.5"][x][0])-1)*100) for x in range(4)]
        except:
            pass

    data_new = {}

    for k,v in data_dict.items():
        data_new[k] = {}
        data_new[k+"0.5"] = {}
        data_new[k]["absolute"] = {}
        data_new[k]["reduction"] = {}
        data_new[k+"0.5"]["absolute"] = {}
        data_new[k+"0.5"]["reduction"] = {}
        for i in range(4):
            data_new[k]["absolute"][str(np.round(data_dict[k]["DF"][i][0]))] = list(data_dict[k]["O"][i])
            data_new[k]["reduction"][str(np.round(data_dict[k]["DF"][i][0]))] = list(data_dict[k]["red"][i])
            try:
                data_new[k+"0.5"]["absolute"][str(np.round(data_dict[k]["DF_y0.5"][i][0]))] = list(data_dict[k]["O_y0.5"][i])
                data_new[k+"0.5"]["reduction"][str(np.round(data_dict[k]["DF_y0.5"][i][0]))] = list(data_dict[k]["red_y0.5"][i])
            except:
                pass

    with open('_qsuite/data_new.json', 'w') as outfile:
        json.dump(data_new, outfile)
def FigS2():
    """
    This function generates Fig. S2
    """
    DF_sw =   np.load('_qsuite/tracing_sim/results_smallworld_DF_NMEAS_100_ONLYSAVETIME_False/results_mean_err.npz')
    DF_exp =  np.load('_qsuite/tracing_sim/results_exponential_DF_NMEAS_100_ONLYSAVETIME_False/results_mean_err.npz')
    exp_R0 =  np.load('_qsuite/tracing_sim/results_exponential_R0_NMEAS_100_ONLYSAVETIME_False/results_mean_err.npz')
    exp_R0_ = np.load("_qsuite/tracing_sim/results_exponential_R0_NMEAS_100_ONLYSAVETIME_False/results.npy")
    sw_R0 =  np.load('_qsuite/tracing_sim/results_smallworld_R0_NMEAS_100_ONLYSAVETIME_False/results_mean_err.npz')
    sw_R0_ = np.load("_qsuite/tracing_sim/results_smallworld_R0_NMEAS_100_ONLYSAVETIME_False/results.npy")
    print(DF_sw["mean"].shape)
    DF = {}
    DF["exp"] = {}
    DF["sw"] = {}
    DF["exp"]["O"] = DF_exp['mean']
    DF["sw"]["O"] =  DF_sw['mean']
    for k,v in DF.items():
        DF[k]["DF"] = (np.array([sum([DF[k]["O"][:,x,i] for i in range(5)]) for x in range(6)])/200_000)/\
                      (np.array([sum([DF[k]["O"][:,x,i] for i in [2,3]]) for x in range(6)])/200_000)
        print(DF[k]["DF"].shape)

    _R0 = np.linspace(0.1,10,51)

    exp_R0_0 = np.array(sum([exp_R0['mean'][0,0,:,i] for i in range(5)]))/200_000
    exp_R0_3 = np.array(sum([exp_R0['mean'][1,0,:,i] for i in range(5)]))/200_000
    exp_R0_var_0 = np.var(np.array(sum([exp_R0_[:,0,0,:,i] for i in range(5)]))/200_000, axis = 0)/exp_R0_0**2
    exp_R0_var_3 = np.var(np.array(sum([exp_R0_[:,1,0,:,i] for i in range(5)]))/200_000, axis = 0)/exp_R0_3**2

    sw_R0_0 = np.array(sum([sw_R0['mean'][0,0,:,i] for i in range(5)]))/200_000
    sw_R0_3 = np.array(sum([sw_R0['mean'][1,0,:,i] for i in range(5)]))/200_000
    sw_R0_var_0 = np.var(np.array(sum([sw_R0_[:,0,0,:,i] for i in range(5)]))/200_000, axis = 0)/sw_R0_0**2
    sw_R0_var_3 = np.var(np.array(sum([sw_R0_[:,1,0,:,i] for i in range(5)]))/200_000, axis = 0)/sw_R0_3**2

    fig, ax = plt.subplots(1,3,figsize = (18,4))

    ax[0].plot(exp_R0_0, color = colors[0])
    ax[0].plot(exp_R0_3, color = colors[1])
    axin0 = ax[0].inset_axes([0.54, 0.1,0.4, 0.4])
    axin0.plot(exp_R0_var_0, color = colors[0])
    axin0.plot(exp_R0_var_3, color = colors[1])

    axin0.set_xticks((0,10,20,30,40,50))
    axin0.set_xticklabels((int(_R0[0]),int(_R0[10]),int(_R0[20]),int(_R0[30]),int(_R0[40]),int(_R0[50])))
    axin0.set_ylabel(r'CV of $\langle\Omega\rangle/N$')
    ax[1].plot(sw_R0_0, color = colors[0])
    ax[1].plot(sw_R0_3, color = colors[1])
    axin1 = ax[1].inset_axes([0.55, 0.1, 0.4, 0.4])
    axin1.plot(sw_R0_var_0, color = colors[0])
    axin1.plot(sw_R0_var_3, color = colors[1])
    axin1.set_xticks((0,10,20,30,40,50))
    axin1.set_xticklabels((int(_R0[0]),int(_R0[10]),int(_R0[20]),int(_R0[30]),int(_R0[40]),int(_R0[50])))
    axin1.set_ylabel(r'CV of $\langle\Omega\rangle/N$')
    lines = [Line2D([0], [0], color = colors[0], alpha = 1, linewidth=1, linestyle='-'),
             Line2D([0], [0], color = colors[0], alpha = 1, linewidth=1, linestyle='-')]

    labels = ['0% app user', '30% app user']

    for i in [0,1]:
        ax[i].set_ylabel(r'$\langle\Omega\rangle/N$')
        ax[i].set_ylim(0,1)
        ax[i].legend(lines,labels)
        ax[i].set_xlabel('$R_0$')
        ax[i].set_xticks((0,10,20,30,40,50))
        ax[i].set_xticklabels((int(_R0[0]),int(_R0[10]),int(_R0[20]),int(_R0[30]),int(_R0[40]),int(_R0[50])))

    for i in range(6):
        ax[2].plot(DF["exp"]["DF"][i],color = colors[i],linewidth=2, ls = '-')
        ax[2].plot(DF["sw"]["DF"][i],color = colors[i],linewidth=2,ls = ':')

    positions2 = (0,6,12,18,24)
    labels2 = (0,25,50,75,100)
    lines2 = [Line2D([0], [0], color = colors[i], alpha = 1, linewidth=2, linestyle='-') for i in [1,2,3,4,5]]
    ylabels2 = ['q = 0.1','q = 0.3','q = 0.5','q = 0.7','q = 0.9']

    ax[2].legend(lines2,ylabels2)
    ax[2].set_xticks(positions2)
    ax[2].set_xticklabels(labels2)
    ax[2].set_ylabel('dark factor DF')
    ax[2].set_xlabel('app participation $a$ [%]')


    plt.show()
FigS2()
def FigS3():
    """
    This function generates Fig. S3
    """
    with open('data_new.json') as json_file:
        data = json.load(json_file)
    fig, axs = plt.subplots(2, 3, figsize = (10,4),sharex=True)
    axss = axs.flatten()

    a = np.linspace(0,24,25)
    xpositions = (0, 6, 12, 18, 24)
    xlabels = ('0%',"25%","50%", "75%",'100%')
    xpositions_ = (0, 12, 24)
    xlabels_ = ('0%',"50%", '100%')
    liste = ["sw","sw_low_eff","sw_noQ","exp","exp_low_eff","exp_noQ"]
    count = 0
    for i in liste:
        k_list = [k_ for k_ in data[i]["reduction"].keys()]
        if len(k_list) !=0:
            ax = axss[count]
            #ax.set_title(i + k_list[1] + k_list[2] + k_list[3])
            count+=1
            axin = ax.inset_axes([0.15, 0.1,0.35, 0.3])

            if count <=3:
                axin.set_ylim(0,0.8)
                axin.set_yticks((0.3,0.7))
            else:
                axin.set_ylim(0.3,0.7)
                axin.set_yticks((0.4,0.6))
            for k_,v_ in data[i]["reduction"].items():
                if k_ == k_list[1] or k_ == k_list[3]:
                    ax.plot(v_,color = "k",  alpha = 1)
                if k_ == k_list[2]:
                    ax.plot(v_,color = "k",  alpha = 1,marker = "o")

            ax.fill_between(a,data[i]["reduction"][k_list[1]],data[i]["reduction"][k_list[3]], color = "grey", alpha = 0.3)

            for k,v in data[i]["absolute"].items():
                if k == k_list[1] or k == k_list[3]:
                    axin.plot(v,color = "k",  alpha = 1)
                if k == k_list[2]:
                    axin.plot(v,color = "k",  alpha = 1,marker = ".")

            axin.fill_between(a,data[i]["absolute"][k_list[1]],data[i]["absolute"][k_list[3]], color = "grey", alpha = 0.3)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            ax.set_xticks(xpositions)
            ax.set_xticklabels(xlabels)

            axss[0].set_ylabel('outbreak size reduction')
            ax.set_xlabel('app participation')
            axin.set_xticks(xpositions_)
            axin.set_xticklabels(xlabels_)
            axin.text(0,1,r'$\langle\Omega\rangle$/N',transform=axin.transAxes,ha='right',va='bottom',**hfont)
    for i in range(3):
        axss[i].set_ylim(-75,0)
    for i in [3,4,5]:
        axss[i].set_ylim(-40,0)
    plt.show()
def FigS4():
    """
    This function generates Fig. S4
    """
    with open('data_new.json') as json_file:
        data = json.load(json_file)
    mean_O_exp_y05 =     data["exp0.5"]["absolute"]
    mean_O_sw_exp_y05 =  data["sw_exp0.5"]["absolute"]

    mean_O_exp =        data["exp"]["absolute"]
    mean_O_sw_exp =     data["sw_exp"]["absolute"]

    red_exp =       data["exp"]["reduction"]
    red_sw_exp =    data["sw_exp"]["reduction"]
    red_exp_y05 =   data["exp0.5"]["reduction"]
    red_sw_exp_y05 = data["sw_exp0.5"]["reduction"]

    fig, ax = plt.subplots(2,2,figsize = (14,10))
    a = np.linspace(0,24,25)


    axin0 = ax[0,0].inset_axes([0.15, 0.1,0.35, 0.3])
    axin1 = ax[0,1].inset_axes([0.15, 0.1,0.35, 0.3])
    axin2 = ax[1,0].inset_axes([0.15, 0.1,0.35, 0.3])
    axin3 = ax[1,1].inset_axes([0.15, 0.1,0.35, 0.3])
    axins = [axin0,axin1,axin2,axin3]
    for i in ["12.0","2.0"]:
        ax[0,0].plot(red_exp[i],color = "k",  alpha = 1)
        ax[0,1].plot(red_exp_y05[i],color = "k",  alpha = 1)
        ax[1,0].plot(red_sw_exp[i],color = "k",  alpha = 1)
        ax[1,1].plot(red_sw_exp_y05[i],color = "k",  alpha = 1)
        axin0.plot(mean_O_exp[i],color = "k",  alpha = 1)
        axin1.plot(mean_O_exp_y05[i],color = "k",  alpha = 1)
        axin2.plot(mean_O_sw_exp[i],color = "k",  alpha = 1)
        axin3.plot(mean_O_sw_exp_y05[i],color = "k",  alpha = 1)

    ax[0,0].plot(red_exp["4.0"],color = "k",  alpha = 1,marker = "o")
    ax[0,1].plot(red_exp_y05["4.0"],color = "k",  alpha = 1,marker = "o")
    ax[1,0].plot(red_sw_exp["4.0"],color = "k",  alpha = 1,marker = "o")
    ax[1,1].plot(red_sw_exp_y05["4.0"],color = "k",  alpha = 1,marker = "o")
    axin0.plot(mean_O_exp["4.0"],color = "k",  alpha = 1,marker = ".")
    axin1.plot(mean_O_exp_y05["4.0"],color = "k",  alpha = 1,marker = ".")
    axin2.plot(mean_O_sw_exp["4.0"],color = "k",  alpha = 1,marker = ".")
    axin3.plot(mean_O_sw_exp_y05["4.0"],color = "k",  alpha = 1,marker = ".")

    ax[0,0].fill_between(a,red_exp["12.0"],red_exp["2.0"], color = "grey", alpha = 0.3)
    ax[0,1].fill_between(a,red_exp_y05["12.0"],red_exp_y05["2.0"], color = "grey", alpha = 0.3)
    ax[1,0].fill_between(a,red_sw_exp["12.0"],red_sw_exp["2.0"], color = "grey", alpha = 0.3)
    ax[1,1].fill_between(a,red_sw_exp_y05["12.0"],red_sw_exp_y05["2.0"], color = "grey", alpha = 0.3)

    axin0.fill_between(a,mean_O_exp["12.0"],mean_O_exp["2.0"], color = "grey", alpha = 0.3)
    axin1.fill_between(a,mean_O_exp_y05["12.0"],mean_O_exp_y05["2.0"], color = "grey", alpha = 0.3)
    axin2.fill_between(a,mean_O_sw_exp["12.0"],mean_O_sw_exp["2.0"], color = "grey", alpha = 0.3)
    axin3.fill_between(a,mean_O_sw_exp_y05["12.0"],mean_O_sw_exp_y05["2.0"], color = "grey", alpha = 0.3)

    positions = (0, 6, 12, 18, 24)
    xlabels = ('0%',"25%","50%", "75%",'100%')

    positionsy1 = (0, -10, -20, -30, -40, -50, -60, -70)
    xlabelsy1 = ('0%', "-10%", "-20%", "-30%", "-40%", "-50%", "-60%", "-70%")

    positionsy = (0, -10, -20, -30, -40)
    xlabelsy = ('0%', "-10%", "-20%", "-30%", "-40%")

    positions1 = (0, 12, 24)
    xlabels1 = ('0%',"50%",'100%')
    for i in [0,1]:
        for j in [0,1]:
            ax[i,j].set_xticks(positions)
            ax[i,j].set_xticklabels(xlabels,**hfont)
            ax[0,i].set_yticks(positionsy)
            ax[0,i].set_yticklabels(xlabelsy,**hfont)
            ax[1,i].set_yticks(positionsy1)
            ax[1,i].set_yticklabels(xlabelsy1,**hfont)
            ax[i,j].set_xlabel(r'app participation $a$',**hfont)
            ax[i,0].set_ylabel(r'outbreak size reduction',**hfont)
            ax[0,i].set_ylim(-40,0)
            ax[1,i].set_ylim(-80,0)

    for i in axins:

        i.set_xticks(positions1)
        i.set_xticklabels(xlabels1,**hfont)
        i.text(0,1,r'$\langle\Omega\rangle$/N',transform=i.transAxes,ha='right',va='bottom',**hfont)
    for i in [axin0,axin1]:
        i.set_ylim(0.3,0.7)
        i.set_yticks((0.4,0.6))
    for i in [axin2,axin3]:
        i.set_ylim(0.1,0.7)
        i.set_yticks((0.3,0.6))
    plt.savefig('Fig6',dpi = 300)
    plt.show()
def FigS6():
    """
    This function generates Fig. S6
    """
    with open('data_new.json') as json_file:
        data = json.load(json_file)
    fig, axs = plt.subplots(1, 2, figsize = (10,4),sharex=True,sharey=True)
    axss = axs.flatten()
    a = np.linspace(0,24,25)
    xpositions = (0, 6, 12, 18, 24)
    xlabels = ('0%',"25%","50%", "75%",'100%')
    xpositions_ = (0, 12, 24)
    xlabels_ = ('0%',"50%", '100%')
    liste = ["sw_exp","sw_exp_"]
    count = 0
    for i in liste:
        k_list = [k_ for k_ in data[i]["reduction"].keys()]
        if len(k_list) !=0:
            ax = axss[count]
            count+=1
            axin = ax.inset_axes([0.15, 0.1,0.35, 0.3])

            for k_,v_ in data[i]["reduction"].items():
                if k_ == k_list[1] or k_ == k_list[3]:
                    ax.plot(v_,color = "k",  alpha = 1)
                if k_ == k_list[2]:
                    ax.plot(v_,color = "k",  alpha = 1,marker = "o")

            ax.fill_between(a,data[i]["reduction"][k_list[1]],data[i]["reduction"][k_list[3]], color = "grey", alpha = 0.3)

            for k,v in data[i]["absolute"].items():
                if k == k_list[1] or k == k_list[3]:
                    axin.plot(v,color = "k",  alpha = 1)
                if k == k_list[2]:
                    axin.plot(v,color = "k",  alpha = 1,marker = ".")

            axin.fill_between(a,data[i]["absolute"][k_list[1]],data[i]["absolute"][k_list[3]], color = "grey", alpha = 0.3)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            ax.set_xticks(xpositions)
            ax.set_xticklabels(xlabels)

            axss[0].set_ylabel('outbreak size reduction')
            ax.set_xlabel('app participation')
            axin.set_xticks(xpositions_)
            axin.set_xticklabels(xlabels_)
            axin.text(0,1,r'$\langle\Omega\rangle$/N',transform=axin.transAxes,ha='right',va='bottom',**hfont)


    plt.show()
