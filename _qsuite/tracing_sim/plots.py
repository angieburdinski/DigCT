import pickle
import matplotlib.pyplot as plt
hfont = {'fontname':'Helvetica'}
plt.rcParams.update({'font.size': 12})
import numpy as np
from matplotlib.lines import Line2D
import gzip
import qsuite_config as cf
import json
import matplotlib.ticker as mtick

def load_export_data():
    exp_noQ = pickle.load(gzip.open('_qsuite/tracing_sim/results_exponential_withoutQ_NMEAS_100_ONLYSAVETIME_False/results.p.gz','rb'))
    exp_noQ = np.array(exp_noQ)
    #lockdown = pickle.load(gzip.open('results_smallworld_lockdown_withQ_NMEAS_100_ONLYSAVETIME_False/results.p.gz','rb'))
    #lockdown = np.array(lockdown)
    #no_lockdown = pickle.load(gzip.open('results_erdosrenyi_withQ_NMEAS_100_ONLYSAVETIME_False/results.p.gz','rb'))
    #no_lockdown = np.array(no_lockdown)

    exp = pickle.load(gzip.open('_qsuite/tracing_sim/results_exponential_withQ_v2_NMEAS_100_ONLYSAVETIME_False/results.p.gz','rb'))
    exp = np.array(exp)
    #sw_exp = pickle.load(gzip.open('results_smallworld_exponential_asc_withQ_NMEAS_100_ONLYSAVETIME_False/results.p.gz','rb'))
    #sw_exp  = np.array(sw_exp )
    #sw_exp_  = pickle.load(gzip.open('results_smallworld_exponential_random_withQ_NMEAS_100_ONLYSAVETIME_False/results.p.gz','rb'))
    #sw_exp_  = np.array(sw_exp_)
    #sw_noQ = pickle.load(gzip.open('results_smallworld_withoutQ_NMEAS_100_ONLYSAVETIME_False/results.p.gz','rb'))
    #sw_noQ = np.array(sw_noQ)
    #sw = pickle.load(gzip.open('results_smallworld_withQ_v3_NMEAS_100_ONLYSAVETIME_False/results.p.gz','rb'))
    #sw = np.array(sw)

    O_exp_no_q =            np.array([sum([exp_noQ[:,:,x,0,i]        for i in range(5)]) for x in range(4)])
    O_exp_base =            np.array([sum([exp[:,:,x,0,i]            for i in range(5)]) for x in range(4)])
    O_exp_y05 =             np.array([sum([exp[:,:,x,1,i]            for i in range(5)]) for x in range(4)])
    #O_sw_base =             [sum([sw[:,:,x,0,i]             for i in range(5)]) for x in range(4)]
    #O_sw_no_q =             [sum([sw_noQ[:,:,x,0,i]         for i in range(5)]) for x in range(4)]
    #O_lockdown_base =       [sum([lockdown[:,:,x,0,i]       for i in range(5)]) for x in range(4)]
    #O_no_lockdown_base =    [sum([no_lockdown[:,:,x,0,i]    for i in range(5)]) for x in range(4)]
    #O_sw_exp_asc =          [sum([sw_exp[:,:,x,0,i]         for i in range(5)]) for x in range(4)]
    #O_sw_exp_ascy05 =       [sum([sw_exp[:,:,x,1,i]         for i in range(5)]) for x in range(4)]
    #O_sw_exp_random =       [sum([sw_exp_[:,:,x,0,i]        for i in range(5)]) for x in range(4)]
    #O_sw_exp_randomy05 =    [sum([sw_exp_[:,:,x,0,i]        for i in range(5)]) for x in range(4)]

    DF_exp_no_q =           (O_exp_no_q)        /np.array([sum([exp_noQ[:,:,x,0,i]      for i in [2,3]]) for x in range(4)])
    DF_exp_base =           (O_exp_base)        /np.array([sum([exp[:,:,x,0,i]          for i in [2,3]]) for x in range(4)])
    DF_exp_y05 =            (O_exp_y05)         /np.array([sum([exp[:,:,x,1,i]          for i in [2,3]]) for x in range(4)])
    #DF_sw_base =            (O_sw_base)         /np.array([sum([sw[:,:,x,0,i]           for i in [2,3]]) for x in range(4)])
    #DF_sw_no_q =            (O_sw_no_q)         /np.array([sum([sw_noQ[:,:,x,0,i]       for i in [2,3]]) for x in range(4)])
    #DF_lockdown_base =      (O_lockdown_base)   /np.array([sum([lockdown[:,:,x,0,i]     for i in [2,3]]) for x in range(4)])
    #DF_no_lockdown_base =   (O_no_lockdown_base)/np.array([sum([no_lockdown[:,:,x,0,i]  for i in [2,3]]) for x in range(4)])
    #DF_sw_exp_asc =         (O_sw_exp_asc)      /np.array([sum([sw_exp[:,:,x,0,i]       for i in [2,3]]) for x in range(4)])
    #DF_sw_exp_ascy05 =      (O_sw_exp_ascy05)   /np.array([sum([sw_exp[:,:,x,1,i]       for i in [2,3]]) for x in range(4)])
    #DF_sw_exp_random =      (O_sw_exp_random)   /np.array([sum([sw_exp_[:,:,x,0,i]      for i in [2,3]]) for x in range(4)])
    #DF_sw_exp_randomy05 =   (O_sw_exp_randomy05)/np.array([sum([sw_exp_[:,:,x,0,i]      for i in [2,3]]) for x in range(4)])

    mean_O_exp_no_q =             np.mean(O_exp_no_q,axis = 1)/200_000
    mean_O_exp_base =             np.mean(O_exp_base,axis = 1)/200_000
    mean_O_exp_y05 =              np.mean(O_exp_y05,axis = 1)/200_000
    #mean_O_sw_base =              np.mean(O_sw_base,axis = 3)/200_000
    #mean_O_sw_no_q =              np.mean(O_sw_no_q,axis = 3)/200_000
    #mean_O_lockdown_base =        np.mean(O_lockdown_base,axis = 3)/200_000
    #mean_O_no_lockdown_base =     np.mean(O_no_lockdown_base,axis = 3)/200_000
    #mean_O_sw_exp_asc =           np.mean(O_sw_exp_asc,axis = 3)/200_000
    #mean_O_sw_exp_ascy05 =        np.mean(O_sw_exp_ascy05,axis = 3)/200_000
    #mean_O_sw_exp_random =        np.mean(O_sw_exp_random,axis = 3)/200_000
    #mean_O_sw_exp_randomy05 =     np.mean(O_sw_exp_randomy05,axis = 3)/200_000

    mean_DF_exp_no_q =            np.mean(DF_exp_no_q,axis = 1)
    mean_DF_exp_base =            np.mean(DF_exp_base,axis = 1)
    mean_DF_exp_y05 =             np.mean(DF_exp_y05,axis = 1)
    #mean_DF_sw_base =             np.mean(DF_sw_base,axis = 3)
    #mean_DF_sw_no_q =             np.mean(DF_sw_no_q,axis = 3)
    #mean_DF_lockdown_base =       np.mean(DF_lockdown_base,axis = 3)
    #mean_DF_no_lockdown_base =    np.mean(DF_no_lockdown_base,axis = 3)
    #mean_DF_sw_exp_asc =          np.mean(DF_sw_exp_asc,axis = 3)
    #mean_DF_sw_exp_ascy05 =       np.mean(DF_sw_exp_ascy05,axis = 3)
    #mean_DF_sw_exp_random =       np.mean(DF_sw_exp_random,axis = 3)
    #mean_DF_sw_exp_randomy05 =    np.mean(DF_sw_exp_randomy05,axis = 3)

    red_exp_no_q =              np.array([(((mean_O_exp_no_q[x]/mean_O_exp_no_q[x][0])-1)*100)                    for x in range(4)])
    red_exp_base =              np.array([(((mean_O_exp_base[x]/mean_O_exp_base[x][0])-1)*100)                    for x in range(4)])
    red_exp_y05 =               np.array([(((mean_O_exp_y05[x]/mean_O_exp_y05[x][0])-1)*100)                      for x in range(4)])
    #red_sw_base =              [(((mean_O_sw_base[x]/mean_O_sw_base[x][0])-1)*100)                      for x in range(4)]
    #red_sw_no_q =              [(((mean_O_sw_no_q[x]/mean_O_sw_no_q[x][0])-1)*100)                      for x in range(4)]
    #red_lockdown_base =        [(((mean_O_lockdown_base[x]/mean_O_lockdown_base[x][0])-1)*100)          for x in range(4)]
    #red_no_lockdown_base =     [(((mean_O_no_lockdown_base[x]/mean_O_no_lockdown_base[x][0])-1)*100)    for x in range(4)]
    #red_sw_exp_asc =           [(((mean_O_sw_exp_asc[x]/mean_O_sw_exp_asc[x][0])-1)*100)                for x in range(4)]
    #red_sw_exp_ascy05 =        [(((mean_O_sw_exp_ascy05[x]/mean_O_sw_exp_ascy05[x][0])-1)*100)          for x in range(4)]
    #red_sw_exp_random =        [(((mean_O_sw_exp_random[x]/mean_O_sw_exp_random[x][0])-1)*100)          for x in range(4)]
    #red_sw_exp_randomy05 =     [(((mean_O_sw_exp_randomy05[x]/mean_O_sw_exp_randomy05[x][0])-1)*100)    for x in range(4)]

    #namelist = ["exp_base","exp_no_q","exp_y05","sw_base","sw_no_q","lockdown_base","no_lockdown_base","sw_exp_asc","red_sw_exp_ascy05","red_sw_exp_random","red_sw_exp_randomy05"]
    namelist = ["exp_base","exp_no_q","exp_y05"]
    data_new = {}
    for i in namelist:
        data_new[i] = {}
        data_new[i]["absolute"] = {}
        data_new[i]["reduction"] = {}
    for i in range(4):
        data_new["exp_base"]["absolute"][str(np.round(mean_DF_exp_base[i][0]))] = list(mean_O_exp_base[i])
        data_new["exp_base"]["reduction"][str(np.round(mean_DF_exp_base[i][0]))] = list(red_exp_base[i])

        data_new["exp_no_q"]["absolute"][str(np.round(mean_DF_exp_no_q[i][0]))] = list(mean_O_exp_no_q[i])
        data_new["exp_no_q"]["reduction"][str(np.round(mean_DF_exp_no_q[i][0]))] = list(red_exp_no_q[i])

        data_new["exp_y05"]["absolute"][str(np.round(mean_DF_exp_y05[i][0]))] = list(mean_O_exp_y05[i])
        data_new["exp_y05"]["reduction"][str(np.round(mean_DF_exp_y05[i][0]))] = list(red_exp_y05[i])

        #data_new["sw_base"]["absolute"][str(np.round(mean_DF_sw_base[i][0],1))] = mean_O_sw_base[i]
        #data_new["sw_base"]["reduction"][str(np.round(mean_DF_sw_base[i][0],1))] = red_sw_base[i]

        #data_new["sw_no_q"]["absolute"][str(np.round(mean_DF_sw_no_q[i][0],1))] = mean_O_sw_no_q[i]
        #data_new["sw_no_q"]["reduction"][str(np.round(mean_DF_sw_no_q[i][0],1))] = red_sw_no_q[i]

        #data_new["lockdown_base"]["absolute"][str(np.round(mean_DF_lockdown_base[i][0],1))] = mean_O_lockdown_base[i]
        #data_new["lockdown_base"]["reduction"][str(np.round(mean_DF_lockdown_base[i][0],1))] = red_lockdown_base[i]

        #data_new["no_lockdown_base"]["absolute"][str(np.round(mean_DF_no_lockdown_base[i][0],1))] = mean_O_no_lockdown_base[i]
        #data_new["no_lockdown_base"]["reduction"][str(np.round(mean_DF_no_lockdown_base[i][0],1))] = red_no_lockdown_base[i]

        #data_new["sw_exp_asc"]["absolute"][str(np.round(mean_DF_sw_exp_asc[i][0],1))] = mean_O_sw_exp_asc[i]
        #data_new["sw_exp_asc"]["reduction"][str(np.round(mean_DF_sw_exp_asc[i][0],1))] = red_sw_exp_asc[i]

        #data_new["red_sw_exp_ascy05"]["absolute"][str(np.round(mean_DF_sw_exp_ascy05[i][0],1))] = mean_O_sw_exp_ascy05[i]
        #data_new["red_sw_exp_ascy05"]["reduction"][str(np.round(mean_DF_sw_exp_ascy05[i][0],1))] = red_sw_exp_ascy05[i]

        #data_new["sw_exp_random"]["absolute"][str(np.round(mean_DF_sw_exp_random[i][0],1))] = mean_O_sw_exp_random[i]
        #data_new["sw_exp_random"]["reduction"][str(np.round(mean_DF_sw_exp_random[i][0],1))] = red_sw_exp_random[i]

        #data_new["sw_exp_randomy05"]["absolute"][str(np.round(mean_DF_sw_exp_randomy05[i][0],1))] = mean_O_sw_exp_randomy05[i]
        #data_new["sw_exp_randomy05"]["reduction"][str(np.round(mean_DF_sw_exp_randomy05[i][0],1))] = red_sw_exp_randomy05[i]

    with open('data_new.json', 'w') as outfile:
        json.dump(data_new, outfile)


def fast_plot(plot_old = True):
    with open('data_new.json') as json_file:
        data = json.load(json_file)
    with open('data_old.json') as json_file:
        data_old = json.load(json_file)
    #namelist = ["exp_base","exp_no_q","exp_y05","sw_base","sw_no_q","lockdown_base","no_lockdown_base","sw_exp_asc","red_sw_exp_ascy05","red_sw_exp_random","red_sw_exp_randomy05"]
    namelist = ["exp_base","exp_no_q","exp_y05"]
    print(data)
    fig, axs = plt.subplots(2, 3, sharex=True,sharey=True)
    axss = axs.flatten()
    a = np.linspace(0,24,25)
    xpositions = (0, 6, 12, 18, 24)
    xlabels = ('0%',"25%","50%", "75%",'100%')
    xpositions_ = (0, 12, 24)
    xlabels_ = ('0%',"50%", '100%')

    for i in namelist:
        ax = axss[namelist.index(i)]
        axin = ax.inset_axes([0.15, 0.1,0.35, 0.3])
        for k_,v_ in data[i]["reduction"].items():
            if k_ == "12.0" or k_ == "2.0":
                ax.plot(v_,color = "k",  alpha = 1)
            if k_ == "4.0":
                ax.plot(v_,color = "k",  alpha = 1,marker = "o")

        ax.fill_between(a,data[i]["reduction"]["12.0"],data[i]["reduction"]["2.0"], color = "grey", alpha = 0.3)

        for k,v in data[i]["absolute"].items():
            if k == "12.0" or k == "2.0":
                axin.plot(v,color = "k",  alpha = 1)
            if k == "4.0":
                axin.plot(v,color = "k",  alpha = 1,marker = ".")

        axin.fill_between(a,data[i]["absolute"]["12.0"],data[i]["absolute"]["2.0"], color = "grey", alpha = 0.3)

        if plot_old == True:

            #ax = axss[namelist.index(i)]
            #axin = ax.inset_axes([0.15, 0.1,0.35, 0.3])
            for k_,v_ in data_old[i]["reduction"].items():
                if k_ == "12.0" or k_ == "2.0":
                    ax.plot(v_,color = "r",  alpha = 0.5)
                if k_ == "4.0":
                    ax.plot(v_,color = "r",  alpha = 0.5, marker = "o")

            ax.fill_between(a,data_old[i]["reduction"]["12.0"],data_old[i]["reduction"]["2.0"], color = "r", alpha = 0.1)

            for k,v in data_old[i]["absolute"].items():
                if k == "12.0" or k == "2.0":
                    axin.plot(v,color = "r",  alpha = 0.5)
                if k == "4.0":
                    axin.plot(v,color = "r",  alpha = 0.5, marker = ".")

            axin.fill_between(a,data_old[i]["absolute"]["12.0"],data_old[i]["absolute"]["2.0"], color = "r", alpha = 0.1)



        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_xticks(xpositions)
        ax.set_xticklabels(xlabels)

        ax.set_ylim(-40,0)
        ax.set_ylabel('outbreak size reduction')
        ax.set_xlabel('app participation')
        axin.set_xticks(xpositions_)
        axin.set_xticklabels(xlabels_)
        axin.text(0,1,r'$\langle\Omega\rangle$/N',transform=axin.transAxes,ha='right',va='bottom',**hfont)
        #ypositions_ = (np.round(min(data[i]["absolute"]["2.0"]),1),np.round(max(data[i]["absolute"]["12.0"]),1))
        ypositions_ = (np.round(min(data[i]["absolute"]["2.0"]),1),0.6)

        axin.set_yticks(ypositions_)
        axss[namelist.index(i)].set_title(i)

    plt.show()

def load_data_old():
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
    sw_exp = pickle.load(gzip.open('/Users/angeliqueburdinski/expotracing/_qsuite/tracing_sim/results_sw_exp_NMEAS_100_ONLYSAVETIME_False/results.p.gz','rb'))
    sw_exp = np.array(sw_exp)
    exp_y05 = pickle.load(gzip.open('/Users/angeliqueburdinski/expotracing/_qsuite/tracing_sim/results_exp_y05_NMEAS_100_ONLYSAVETIME_False/results.p.gz','rb'))
    exp_y05 = np.array(exp_y05)
    sw_exp_y05 = pickle.load(gzip.open('/Users/angeliqueburdinski/expotracing/_qsuite/tracing_sim/results_sw_exp_y05_NMEAS_100_ONLYSAVETIME_False/results.p.gz','rb'))
    sw_exp_y05 = np.array(sw_exp_y05)

    O_exp =         [sum([exp[:,0,1,:,x,i]          for i in [2,3,4,5,6]]) for x in range(4)]
    O_sw =          [sum([sw[:,0,1,:,x,i]           for i in [2,3,4,5,6]]) for x in range(4)]
    O_lockdown =    [sum([lockdown[:,0,0,:,x,i]     for i in [2,3,4,5,6]]) for x in range(4)]
    O_no_lockdown = [sum([no_lockdown[:,0,0,:,x,i]  for i in [2,3,4,5,6]]) for x in range(4)]
    O_sw_exp =      [sum([sw_exp[:,0,0,:,x,i]       for i in [0,1,2,3,4]]) for x in range(4)]
    O_exp_low_eff = [sum([exp[:,0,0,:,x,i]          for i in [2,3,4,5,6]]) for x in range(4)]
    O_exp_no_q =    [sum([exp_noQ[:,0,0,:,x,i]      for i in [2,3,4,5,6]]) for x in range(4)]
    O_sw_low_eff =  [sum([sw[:,0,0,:,x,i]           for i in [2,3,4,5,6]]) for x in range(4)]
    O_sw_no_q =     [sum([sw_noQ[:,0,0,:,x,i]       for i in [2,3,4,5,6]]) for x in range(4)]
    O_exp_y05 =     [sum([exp_y05[:,0,0,:,x,i]      for i in [0,1,2,3,4]]) for x in range(4)]
    O_sw_exp_y05 =  [sum([sw_exp_y05[:,0,0,:,x,i]   for i in [0,1,2,3,4]]) for x in range(4)]

    DF_exp =         (O_exp)/np.array([sum([exp[:,0,1,:,x,i]                    for i in [4,5]]) for x in range(4)])
    DF_sw =          (O_sw)/np.array([sum([sw[:,0,1,:,x,i]                      for i in [4,5]]) for x in range(4)])
    DF_lockdown =    (O_lockdown)/np.array([sum([lockdown[:,0,0,:,x,i]          for i in [4,5]]) for x in range(4)])
    DF_no_lockdown = (O_no_lockdown)/np.array([sum([no_lockdown[:,0,0,:,x,i]    for i in [4,5]]) for x in range(4)])
    DF_sw_exp =      (O_sw_exp)/np.array([sum([sw_exp[:,0,0,:,x,i]              for i in [2,3]]) for x in range(4)])
    DF_exp_low_eff = (O_exp_low_eff)/np.array([sum([exp[:,0,1,:,x,i]            for i in [4,5]]) for x in range(4)])
    DF_exp_no_q =    (O_exp_no_q)/np.array([sum([exp[:,0,1,:,x,i]               for i in [4,5]]) for x in range(4)])
    DF_sw_low_eff =  (O_sw_low_eff)/np.array([sum([sw[:,0,1,:,x,i]              for i in [4,5]]) for x in range(4)])
    DF_sw_no_q =     (O_sw_no_q)/np.array([sum([sw[:,0,1,:,x,i]                 for i in [4,5]]) for x in range(4)])
    DF_exp_y05 =     (O_exp_y05)/np.array([sum([exp_y05[:,0,0,:,x,i]            for i in [2,3]]) for x in range(4)])
    DF_sw_exp_y05 =  (O_sw_exp_y05)/np.array([sum([sw_exp_y05[:,0,0,:,x,i]      for i in [2,3]]) for x in range(4)])


    mean_O_exp =         np.mean(O_exp,axis = 1)/200_000
    mean_O_sw =          np.mean(O_sw,axis = 1)/200_000
    mean_O_lockdown =    np.mean(O_lockdown,axis = 1)/200_000
    mean_O_no_lockdown = np.mean(O_no_lockdown,axis = 1)/200_000
    mean_O_sw_exp =      np.mean(O_sw_exp,axis = 1)/200_000
    mean_O_exp_low_eff = np.mean(O_exp_low_eff,axis = 1)/200_000
    mean_O_exp_no_q =    np.mean(O_exp_no_q,axis = 1)/200_000
    mean_O_sw_low_eff =  np.mean(O_sw_low_eff,axis = 1)/200_000
    mean_O_sw_no_q =     np.mean(O_sw_no_q,axis = 1)/200_000
    mean_O_exp_y05 =     np.mean( O_exp_y05,axis = 1)/200_000
    mean_O_sw_exp_y05 =  np.mean( O_sw_exp_y05,axis = 1)/200_000

    mean_DF_exp =         np.mean(DF_exp,axis = 1)
    mean_DF_sw =          np.mean(DF_sw,axis = 1)
    mean_DF_lockdown =    np.mean(DF_lockdown,axis = 1)
    mean_DF_no_lockdown = np.mean(DF_no_lockdown,axis = 1)
    mean_DF_sw_exp =      np.mean(DF_sw_exp,axis = 1)
    mean_DF_exp_low_eff = np.mean(DF_exp_low_eff,axis = 1)
    mean_DF_exp_no_q =    np.mean(DF_exp_no_q,axis = 1)
    mean_DF_sw_low_eff =  np.mean(DF_sw_low_eff,axis = 1)
    mean_DF_sw_no_q =     np.mean(DF_sw_no_q,axis = 1)
    mean_DF_exp_y05  =    np.mean(DF_exp_y05,axis = 1)
    mean_DF_sw_exp_y05 =  np.mean(DF_sw_exp_y05,axis = 1)

    red_exp =        [(((mean_O_exp[x]/mean_O_exp[x][0])-1)*100)                    for x in range(4)]
    red_sw =         [(((mean_O_sw[x]/mean_O_sw[x][0])-1)*100)                      for x in range(4)]
    red_lockdown =   [(((mean_O_lockdown[x]/mean_O_lockdown[x][0])-1)*100)          for x in range(4)]
    red_no_lockdown =[(((mean_O_no_lockdown[x]/mean_O_no_lockdown[x][0])-1)*100)    for x in range(4)]
    red_sw_exp =     [(((mean_O_sw_exp[x]/mean_O_sw_exp[x][0])-1)*100)              for x in range(4)]
    red_exp_low_eff =[(((mean_O_exp_low_eff[x]/mean_O_exp_low_eff[x][0])-1)*100)    for x in range(4)]
    red_exp_no_q =   [(((mean_O_exp_no_q[x]/mean_O_exp_no_q[x][0])-1)*100)          for x in range(4)]
    red_sw_low_eff = [(((mean_O_sw_low_eff[x]/mean_O_sw_low_eff[x][0])-1)*100)      for x in range(4)]
    red_sw_no_q =    [(((mean_O_sw_no_q[x]/mean_O_sw_no_q[x][0])-1)*100)            for x in range(4)]
    red_exp_y05 =    [(((mean_O_exp_y05[x]/mean_O_exp_y05[x][0])-1)*100)            for x in range(4)]
    red_sw_exp_y05 = [(((mean_O_sw_exp_y05[x]/mean_O_sw_exp_y05[x][0])-1)*100)      for x in range(4)]

    namelist = ["exp_base","exp_no_q","exp_y05","sw_base","sw_no_q","lockdown_base","no_lockdown_base","sw_exp_asc","red_sw_exp_ascy05"]

    data_old = {}
    for i in namelist:
        data_old[i] = {}
        data_old[i]["absolute"] = {}
        data_old[i]["reduction"] = {}
    for i in range(4):
        data_old["exp_base"]["absolute"][str(np.round(mean_DF_exp[i][0]))] = list(mean_O_exp[i])
        data_old["exp_base"]["reduction"][str(np.round(mean_DF_exp[i][0]))] = list(red_exp[i])

        data_old["exp_no_q"]["absolute"][str(np.round(mean_DF_exp_no_q[i][0]))] = list(mean_O_exp_no_q[i])
        data_old["exp_no_q"]["reduction"][str(np.round(mean_DF_exp_no_q[i][0]))] = list(red_exp_no_q[i])

        data_old["exp_y05"]["absolute"][str(np.round(mean_DF_exp_y05[i][0]))] = list(mean_O_exp_y05[i])
        data_old["exp_y05"]["reduction"][str(np.round(mean_DF_exp_y05[i][0]))] = list(red_exp_y05[i])

        data_old["sw_base"]["absolute"][str(np.round(mean_DF_sw[i][0],1))] = list(mean_O_sw[i])
        data_old["sw_base"]["reduction"][str(np.round(mean_DF_sw[i][0],1))] = list(red_sw[i])

        data_old["sw_no_q"]["absolute"][str(np.round(mean_DF_sw_no_q[i][0],1))] = list(mean_O_sw_no_q[i])
        data_old["sw_no_q"]["reduction"][str(np.round(mean_DF_sw_no_q[i][0],1))] = list(red_sw_no_q[i])

        data_old["lockdown_base"]["absolute"][str(np.round(mean_DF_lockdown[i][0],1))] = list(mean_O_lockdown[i])
        data_old["lockdown_base"]["reduction"][str(np.round(mean_DF_lockdown[i][0],1))] = list(red_lockdown[i])

        data_old["no_lockdown_base"]["absolute"][str(np.round(mean_DF_no_lockdown[i][0],1))] = list(mean_O_no_lockdown[i])
        data_old["no_lockdown_base"]["reduction"][str(np.round(mean_DF_no_lockdown[i][0],1))] = list(red_no_lockdown[i])

        data_old["sw_exp_asc"]["absolute"][str(np.round(mean_DF_sw_exp[i][0],1))] = list(mean_O_sw_exp[i])
        data_old["sw_exp_asc"]["reduction"][str(np.round(mean_DF_sw_exp[i][0],1))] = list(red_sw_exp[i])

        data_old["red_sw_exp_ascy05"]["absolute"][str(np.round(mean_DF_sw_exp_y05[i][0],1))] = list(mean_O_sw_exp_y05[i])
        data_old["red_sw_exp_ascy05"]["reduction"][str(np.round(mean_DF_sw_exp_y05[i][0],1))] = list(red_sw_exp_y05[i])

    with open('data_old.json', 'w') as outfile:
        json.dump(data_old, outfile)
#load_data_old()
fast_plot(plot_old = True)





def R_0():
    R0 = pickle.load(gzip.open('/Users/angeliqueburdinski/expotracing/_qsuite/tracing_sim/results_R0_NMEAS_100_ONLYSAVETIME_False/results.p.gz','rb'))
    R0 = np.array(R0)

    R0sw = pickle.load(gzip.open('/Users/angeliqueburdinski/expotracing/_qsuite/tracing_sim/results_R0sw_NMEAS_100_ONLYSAVETIME_False/results.p.gz','rb'))
    R0sw = np.array(R0sw)
    _R0 = np.linspace(0.1,10,51)
    mean0 = np.mean(R0[:,:,0,0,0], axis = 0)/200_000
    mean3 = np.mean(R0[:,:,0,1,0], axis = 0)/200_000
    var0 =  np.var(R0[:,:,0,0,0]/200_000,  axis = 0)/mean0**2
    var3 =  np.var(R0[:,:,0,1,0]/200_000,  axis = 0)/mean3**2

    mean0sw = np.mean(R0sw[:,:,0,0,0], axis = 0)/200_000
    mean3sw = np.mean(R0sw[:,:,0,1,0], axis = 0)/200_000
    var0sw =  np.var(R0sw[:,:,0,0,0]/200_000,  axis = 0)/mean0sw**2
    var3sw =  np.var(R0sw[:,:,0,1,0]/200_000,  axis = 0)/mean3sw**2

    fig, ax = plt.subplots(1,3,figsize = (18,5))

    ax[0].plot(mean0, color = colors[0])
    ax[0].plot(mean3, color = colors[1])
    axin0 = ax[0].inset_axes([0.54, 0.1,0.4, 0.4])
    axin0.plot(var0,  color = colors[0])
    axin0.plot(var3,  color = colors[1])

    axin0.set_xticks((0,10,20,30,40,50))
    axin0.set_xticklabels((int(_R0[0]),int(_R0[10]),int(_R0[20]),int(_R0[30]),int(_R0[40]),int(_R0[50])))
    axin0.set_ylabel(r'CV of $\langle\Omega\rangle/N$')
    ax[1].plot(mean0sw, color = colors[0])
    ax[1].plot(mean3sw, color = colors[1])
    axin1 = ax[1].inset_axes([0.55, 0.1, 0.4, 0.4])
    axin1.plot(var0sw,  color = colors[0])
    axin1.plot(var3sw,  color = colors[1])
    axin1.set_xticks((0,10,20,30,40,50))
    axin1.set_xticklabels((int(_R0[0]),int(_R0[10]),int(_R0[20]),int(_R0[30]),int(_R0[40]),int(_R0[50])))
    axin1.set_ylabel(r'CV of $\langle\Omega\rangle/N$')
    lines = [Line2D([0], [0], color = colors[0], alpha = 1, linewidth=1, linestyle='-'),
             Line2D([0], [0], color = colors[1], alpha = 1, linewidth=1, linestyle='-')]
    labels = ['0% app user', '30% app user']

    for i in [0,1]:
        ax[i].set_ylabel(r'$\langle\Omega\rangle/N$')
        ax[i].set_ylim(0,1)
        ax[i].legend(lines,labels)
        ax[i].set_xlabel('$R_0$')
        ax[i].set_xticks((0,10,20,30,40,50))
        ax[i].set_xticklabels((int(_R0[0]),int(_R0[10]),int(_R0[20]),int(_R0[30]),int(_R0[40]),int(_R0[50])))

    for i in range(6):
        ax[2].plot(mean_DF_exp[i], color = colors[i],linewidth=2, ls = '-')
        ax[2].plot(mean_DF_sw[i], color = colors[i],linewidth=2,ls = ':')
        #ax[2].plot(mean_DF_lockdown[i], color = colors[i],linewidth=2)
        #ax[2].plot(mean_DF_no_lockdown[i], color = colors[i],linewidth=2)
        #ax[2].plot(mean_DF_sw_exp[i], color = colors[i],linewidth=2)

    positions2 = (0,6,12,18,24)
    labels2 = (0,25,50,75,100)
    lines2 = [Line2D([0], [0], color = colors[i], alpha = 1, linewidth=2, linestyle='-') for i in [1,2,3,4,5]]
    ylabels2 = ['q = 0.1','q = 0.3','q = 0.5','q = 0.7','q = 0.9']

    ax[2].set_xticks(positions2)
    ax[2].set_xticklabels(labels2)
    ax[2].set_ylabel('dark factor DF')
    ax[2].set_xlabel('app participation $a$ [%]')
    ax[2].legend(lines2,ylabels2)

    plt.savefig('R0',dpi=300)
    plt.show()

def Figure4():
    O_exp =         [sum([exp[:,0,1,:,x,i] for i in [2,3,4,5,6]]) for x in range(len(cf.q))]
    O_exp_low_eff = [sum([exp[:,0,0,:,x,i] for i in [2,3,4,5,6]]) for x in range(len(cf.q))]
    O_exp_no_q =    [sum([exp_noQ[:,0,0,:,x,i] for i in [2,3,4,5,6]]) for x in range(len(cf.q))]

    DF_exp =         (O_exp)/np.array([sum([exp[:,0,1,:,x,i] for i in [4,5]]) for x in range(len(cf.q))])
    DF_exp_low_eff = (O_exp_low_eff)/np.array([sum([exp[:,0,1,:,x,i] for i in [4,5]]) for x in range(len(cf.q))])
    DF_exp_no_q =    (O_exp_no_q)/np.array([sum([exp[:,0,1,:,x,i] for i in [4,5]]) for x in range(len(cf.q))])

    mean_O_exp =         np.mean(O_exp,axis = 1)/200_000
    mean_O_exp_low_eff = np.mean(O_exp_low_eff,axis = 1)/200_000
    mean_O_exp_no_q =    np.mean(O_exp_no_q,axis = 1)/200_000

    mean_DF_exp =         np.mean(DF_exp,axis = 1)
    mean_DF_exp_low_eff = np.mean(DF_exp_low_eff,axis = 1)
    mean_DF_exp_no_q =    np.mean(DF_exp_no_q,axis = 1)

    red_exp =         [(((mean_O_exp[x]/mean_O_exp[x][0])-1)*100) for x in range(6)]
    red_exp_low_eff = [(((mean_O_exp_low_eff[x]/mean_O_exp_low_eff[x][0])-1)*100) for x in range(6)]
    red_exp_no_q =    [(((mean_O_exp_no_q[x]/mean_O_exp_no_q[x][0])-1)*100) for x in range(6)]

    fig, ax = plt.subplots(1,3,figsize = (18,5))
    a = np.linspace(0,24,25)
    axin0 = ax[0].inset_axes([0.15, 0.1,0.35, 0.3])
    axin1 = ax[1].inset_axes([0.15, 0.1,0.35, 0.3])
    axin2 = ax[2].inset_axes([0.15, 0.1,0.35, 0.3])

    for i in [1,3]:
        ax[0].plot(red_exp[i],color = "k",  alpha = 1)
        ax[1].plot(red_exp_low_eff[i],color = "k",  alpha = 1)
        ax[2].plot(red_exp_no_q[i],color = "k",  alpha = 1)
        axin0.plot(mean_O_exp[i],color = "k",  alpha = 1)
        axin1.plot(mean_O_exp_low_eff[i],color = "k",  alpha = 1)
        axin2.plot(mean_O_exp_no_q[i],color = "k",  alpha = 1)

    ax[0].plot(red_exp[2],color = "k",  alpha = 1,marker = "o")
    ax[1].plot(red_exp_low_eff[2],color = "k",  alpha = 1,marker = "o")
    ax[2].plot(red_exp_no_q[2],color = "k",  alpha = 1,marker = "o")
    axin0.plot(mean_O_exp[2],color = "k",  alpha = 1,marker = ".")
    axin1.plot(mean_O_exp_low_eff[2],color = "k",  alpha = 1,marker = ".")
    axin2.plot(mean_O_exp_no_q[2],color = "k",  alpha = 1,marker = ".")

    ax[0].fill_between(a,red_exp[1],red_exp[3], color = "grey", alpha = 0.3)
    ax[1].fill_between(a,red_exp_low_eff[1],red_exp_low_eff[3], color = "grey", alpha = 0.3)
    ax[2].fill_between(a,red_exp_no_q[1],red_exp_no_q[3], color = "grey", alpha = 0.3)

    axin0.fill_between(a,mean_O_exp[1],mean_O_exp[3], color = "grey", alpha = 0.3)
    axin1.fill_between(a,mean_O_exp_low_eff[1],mean_O_exp_low_eff[3], color = "grey", alpha = 0.3)
    axin2.fill_between(a,mean_O_exp_no_q[1],mean_O_exp_no_q[3], color = "grey", alpha = 0.3)

    positions = (0, 6, 12, 18, 24)
    xlabels = ('0%',"25%","50%", "75%",'100%')

    positionsy = (0, -10, -20, -30, -40,)
    xlabelsy = ('0%', "-10%", "-20%", "-30%", "-40%",)

    positions1 = (0, 12, 24)
    xlabels1 = ('0%',"50%",'100%')
    for i in [0,1,2]:
        ax[i].set_xticks(positions)
        ax[i].set_xticklabels(xlabels,**hfont)
        ax[i].set_yticks(positionsy)
        ax[i].set_yticklabels(xlabelsy,**hfont)
        ax[i].set_xlabel(r'app participation $a$',**hfont)
        ax[0].set_ylabel(r'outbreak size reduction',**hfont)
        ax[i].set_ylim(-40,0)
    axins = [axin0,axin1,axin2]
    for i in axins:
        i.set_ylim(0.3,0.7)
        i.set_yticks((0.4,0.6))
        i.set_xticks(positions1)
        i.set_xticklabels(xlabels1,**hfont)
        i.text(0,1,r'$\langle\Omega\rangle$/N',transform=i.transAxes,ha='right',va='bottom',**hfont)

    plt.savefig('Fig4',dpi = 300)
    plt.show()

def Figure5():
    O_sw =         [sum([sw[:,0,1,:,x,i] for i in [2,3,4,5,6]]) for x in range(len(cf.q))]
    O_sw_low_eff = [sum([sw[:,0,0,:,x,i] for i in [2,3,4,5,6]]) for x in range(len(cf.q))]
    O_sw_no_q =    [sum([sw_noQ[:,0,0,:,x,i] for i in [2,3,4,5,6]]) for x in range(len(cf.q))]

    DF_sw =         (O_sw)/np.array([sum([sw[:,0,1,:,x,i] for i in [4,5]]) for x in range(len(cf.q))])
    DF_sw_low_eff = (O_sw_low_eff)/np.array([sum([sw[:,0,1,:,x,i] for i in [4,5]]) for x in range(len(cf.q))])
    DF_sw_no_q =    (O_sw_no_q)/np.array([sum([sw[:,0,1,:,x,i] for i in [4,5]]) for x in range(len(cf.q))])

    mean_O_sw =         np.mean(O_sw,axis = 1)/200_000
    mean_O_sw_low_eff = np.mean(O_sw_low_eff,axis = 1)/200_000
    mean_O_sw_no_q =    np.mean(O_sw_no_q,axis = 1)/200_000

    mean_DF_sw =         np.mean(DF_sw,axis = 1)
    mean_DF_sw_low_eff = np.mean(DF_sw_low_eff,axis = 1)
    mean_DF_sw_no_q =    np.mean(DF_sw_no_q,axis = 1)

    red_sw =         [(((mean_O_sw[x]/mean_O_sw[x][0])-1)*100) for x in range(6)]
    red_sw_low_eff = [(((mean_O_sw_low_eff[x]/mean_O_sw_low_eff[x][0])-1)*100) for x in range(6)]
    red_sw_no_q =    [(((mean_O_sw_no_q[x]/mean_O_sw_no_q[x][0])-1)*100) for x in range(6)]

    fig, ax = plt.subplots(1,3,figsize = (18,5))
    a = np.linspace(0,24,25)
    axin0 = ax[0].inset_axes([0.15, 0.1,0.35, 0.3])
    axin1 = ax[1].inset_axes([0.15, 0.1,0.35, 0.3])
    axin2 = ax[2].inset_axes([0.15, 0.1,0.35, 0.3])

    for i in [1,3]:
        ax[0].plot(red_sw[i],color = "k",  alpha = 1)
        ax[1].plot(red_sw_low_eff[i],color = "k",  alpha = 1)
        ax[2].plot(red_sw_no_q[i],color = "k",  alpha = 1)
        axin0.plot(mean_O_sw[i],color = "k",  alpha = 1)
        axin1.plot(mean_O_sw_low_eff[i],color = "k",  alpha = 1)
        axin2.plot(mean_O_sw_no_q[i],color = "k",  alpha = 1)

    ax[0].plot(red_sw[2],color = "k",  alpha = 1,marker = "o")
    ax[1].plot(red_sw_low_eff[2],color = "k",  alpha = 1,marker = "o")
    ax[2].plot(red_sw_no_q[2],color = "k",  alpha = 1,marker = "o")
    axin0.plot(mean_O_sw[2],color = "k",  alpha = 1,marker = ".")
    axin1.plot(mean_O_sw_low_eff[2],color = "k",  alpha = 1,marker = ".")
    axin2.plot(mean_O_sw_no_q[2],color = "k",  alpha = 1,marker = ".")

    ax[0].fill_between(a,red_sw[1],red_sw[3], color = "grey", alpha = 0.3)
    ax[1].fill_between(a,red_sw_low_eff[1],red_sw_low_eff[3], color = "grey", alpha = 0.3)
    ax[2].fill_between(a,red_sw_no_q[1],red_sw_no_q[3], color = "grey", alpha = 0.3)

    axin0.fill_between(a,mean_O_sw[1],mean_O_sw[3], color = "grey", alpha = 0.3)
    axin1.fill_between(a,mean_O_sw_low_eff[1],mean_O_sw_low_eff[3], color = "grey", alpha = 0.3)
    axin2.fill_between(a,mean_O_sw_no_q[1],mean_O_sw_no_q[3], color = "grey", alpha = 0.3)

    positions = (0, 6, 12, 18, 24)
    xlabels = ('0%',"25%","50%", "75%",'100%')

    positionsy = (0, -10, -20, -30, -40, -50, -60, -70)
    xlabelsy = ('0%', "-10%", "-20%", "-30%", "-40%", "-50%", "-60%", "-70%")

    positions1 = (0, 12, 24)
    xlabels1 = ('0%',"50%",'100%')
    for i in [0,1,2]:
        ax[i].set_xticks(positions)
        ax[i].set_xticklabels(xlabels,**hfont)
        ax[i].set_yticks(positionsy)
        ax[i].set_yticklabels(xlabelsy,**hfont)
        ax[i].set_xlabel(r'app participation $a$',**hfont)
        ax[0].set_ylabel(r'outbreak size reduction',**hfont)
        ax[i].set_ylim(-75,0)
    axins = [axin0,axin1,axin2]
    for i in axins:
        i.set_ylim(0,0.8)
        i.set_yticks((0.3,0.7))
        i.set_xticks(positions1)
        i.set_xticklabels(xlabels1,**hfont)
        i.text(0,1,r'$\langle\Omega\rangle$/N',transform=i.transAxes,ha='right',va='bottom',**hfont)

    plt.savefig('Fig5',dpi = 300)
    plt.show()


def Figure6():
    O_exp =             [sum([exp[:,0,1,:,x,i] for i in [2,3,4,5,6]]) for x in range(len(cf.q))]
    O_sw_exp =          [sum([sw_exp[:,0,0,:,x,i] for i in [0,1,2,3,4]]) for x in range(len(cf.q))]
    O_exp_y05 =         [sum([exp_y05[:,0,0,:,x,i] for i in [0,1,2,3,4]]) for x in range(len(cf.q))]
    O_sw_exp_y05 =      [sum([sw_exp_y05[:,0,0,:,x,i] for i in [0,1,2,3,4]]) for x in range(len(cf.q))]

    mean_O_exp_y05 =         np.mean( O_exp_y05,axis = 1)/200_000
    mean_O_sw_exp_y05 =      np.mean( O_sw_exp_y05,axis = 1)/200_000

    mean_O_exp =         np.mean( O_exp,axis = 1)/200_000
    mean_O_sw_exp =      np.mean( O_sw_exp,axis = 1)/200_000

    red_exp =        [(((mean_O_exp[x]/mean_O_exp[x][0])-1)*100) for x in range(6)]
    red_sw_exp =     [(((mean_O_sw_exp[x]/mean_O_sw_exp[x][0])-1)*100) for x in range(6)]
    red_exp_y05 =    [(((mean_O_exp_y05[x]/mean_O_exp_y05[x][0])-1)*100) for x in range(6)]
    red_sw_exp_y05 = [(((mean_O_sw_exp_y05[x]/mean_O_sw_exp_y05[x][0])-1)*100) for x in range(6)]

    fig, ax = plt.subplots(2,2,figsize = (14,10))
    a = np.linspace(0,24,25)


    axin0 = ax[0,0].inset_axes([0.15, 0.1,0.35, 0.3])
    axin1 = ax[0,1].inset_axes([0.15, 0.1,0.35, 0.3])
    axin2 = ax[1,0].inset_axes([0.15, 0.1,0.35, 0.3])
    axin3 = ax[1,1].inset_axes([0.15, 0.1,0.35, 0.3])
    axins = [axin0,axin1,axin2,axin3]
    for i in [1,3]:
        ax[0,0].plot(red_exp[i],color = "k",  alpha = 1)
        ax[0,1].plot(red_exp_y05[i],color = "k",  alpha = 1)
        ax[1,0].plot(red_sw_exp[i],color = "k",  alpha = 1)
        ax[1,1].plot(red_sw_exp_y05[i],color = "k",  alpha = 1)
        axin0.plot(mean_O_exp[i],color = "k",  alpha = 1)
        axin1.plot(mean_O_exp_y05[i],color = "k",  alpha = 1)
        axin2.plot(mean_O_sw_exp[i],color = "k",  alpha = 1)
        axin3.plot(mean_O_sw_exp_y05[i],color = "k",  alpha = 1)

    ax[0,0].plot(red_exp[2],color = "k",  alpha = 1,marker = "o")
    ax[0,1].plot(red_exp_y05[2],color = "k",  alpha = 1,marker = "o")
    ax[1,0].plot(red_sw_exp[2],color = "k",  alpha = 1,marker = "o")
    ax[1,1].plot(red_sw_exp_y05[2],color = "k",  alpha = 1,marker = "o")
    axin0.plot(mean_O_exp[2],color = "k",  alpha = 1,marker = ".")
    axin1.plot(mean_O_exp_y05[2],color = "k",  alpha = 1,marker = ".")
    axin2.plot(mean_O_sw_exp[2],color = "k",  alpha = 1,marker = ".")
    axin3.plot(mean_O_sw_exp_y05[2],color = "k",  alpha = 1,marker = ".")

    ax[0,0].fill_between(a,red_exp[1],red_exp[3], color = "grey", alpha = 0.3)
    ax[0,1].fill_between(a,red_exp_y05[1],red_exp_y05[3], color = "grey", alpha = 0.3)
    ax[1,0].fill_between(a,red_sw_exp[1],red_sw_exp[3], color = "grey", alpha = 0.3)
    ax[1,1].fill_between(a,red_sw_exp_y05[1],red_sw_exp_y05[3], color = "grey", alpha = 0.3)

    axin0.fill_between(a,mean_O_exp[1],mean_O_exp[3], color = "grey", alpha = 0.3)
    axin1.fill_between(a,mean_O_exp_y05[1],mean_O_exp_y05[3], color = "grey", alpha = 0.3)
    axin2.fill_between(a,mean_O_sw_exp[1],mean_O_sw_exp[3], color = "grey", alpha = 0.3)
    axin3.fill_between(a,mean_O_sw_exp_y05[1],mean_O_sw_exp_y05[3], color = "grey", alpha = 0.3)

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
