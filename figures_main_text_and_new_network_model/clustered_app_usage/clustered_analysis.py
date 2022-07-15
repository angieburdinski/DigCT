import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
plt.rcParams.update({'font.size': 8,
                    'font.sans-serif': 'Helvetica'})
p1 = Path(__file__).parents[2]
p2 = Path(__file__).parents[0]
lss = ['solid','dashed']
colors = ['#333333','#888888']
clus = ['True','False']
res_keys = ['app','counts']
N_sim  = 200_000
labels = ['clustered app participation','random app participation']
colors= [
      '#333333',
      '#5540BF',
      '#FE6100',
      '#FFB000',
      ]
networks = ['ER','SW','ER_exp','SW_exp']

def clustered_app_old_version():
    network_keys = ['sw','sw_exp']
    data = {}
    for i_n, n in enumerate(network_keys):
        data[n] = np.load(str(p1)+'/analysis_collection/tracing_sim/results_'+str(n)\
                            +'_clustered/results_mean_err.npz')['mean']

    calc = {}
    calc['a']= np.linspace(0,1,10)*100

    for i_n, n in enumerate(network_keys):
        calc[n] = {}
        for i_c, c in enumerate(clus):
            calc[n][c] = {}
            O   = np.array([sum([data[n][i_c,:,i] for i in range(5)])]).flatten()/N_sim
            DF  = O/(np.array([sum([data[n][i_c,:,i] for i in [2,3]])]).flatten()/N_sim)
            red = ((O/O[0])-1)*100
            print(str(np.round(DF[0])))
            calc[n][c] = list(red)

    contact_calc = {}
    contacts = np.load(str(p2)+'/results_clustered_found_contacts_test_v2/results.npy',allow_pickle=True)

    contact_calc['a']  = np.linspace(0.001,0.999,10)*100
    for i_n, n in enumerate(network_keys):
        contact_calc[n] = {}
        for i_c, c in enumerate(reversed(clus)): #we do reversed here as there is a mistake in the respective simulation.py
            contact_calc[n][eval(c)] = {}
            for i_r, r in enumerate(res_keys):
                x = contacts[0,:,i_n,i_c]
                contact_calc[n][eval(c)][r] = x[:,i_r]

    fig, axs = plt.subplots(2,3,sharey = "col")

    for i_n, n in enumerate(network_keys):
        for i_c, c in enumerate(clus):

            axs[i_n,0].plot(calc['a'], calc[n][c], marker = 'o', color = colors[i_c], ls = lss[i_c],\
                            label = labels[i_c])
            axs[i_n,0].yaxis.set_major_formatter(mtick.PercentFormatter())

            c = eval(c)

            for i_r, r in enumerate(res_keys):
                if r == 'counts':
                    axs[i_n,1].plot(contact_calc['a'], contact_calc[n][c][r]/(N_sim*10),\
                    label = labels[i_c], marker = 'o', color = colors[i_c], ls = lss[i_c])

                if r == 'app':
                    print(contact_calc['a'][3])
                    axs[i_n,2].hist(contact_calc[n][c][r][3], bins = np.arange(0,N_sim+N_sim/100,N_sim/100),\
                    label = labels[i_c],alpha = 0.6)

            axs[i_n,i_c].spines['bottom'].set_visible(False)
            axs[i_n,i_c].spines['right'].set_visible(False)
            axs[i_n,i_c].xaxis.tick_top()
            axs[0,i_c].text(1.05,1.25,'app participation $a$ ',transform=axs[0,i_c].transAxes,va='top',ha='right')

        for i in range(2):
            axs[i_n,i].xaxis.set_major_formatter(mtick.PercentFormatter())

        axs[0,0].legend()
        axs[i_n,0].set_ylabel(r'outbreak size reduction' ,loc='bottom')
        axs[i_n,1].set_ylabel(r'fraction found contacts' ,loc='bottom')
        axs[i_n,2].set_ylabel(r'app participants' ,loc='bottom')
        axs[1,2].set_xlabel('node index')
        axs[0,2].legend()


    plt.show()

def get_data_calculations():
    res_keys = ['counts']

    a = np.linspace(0,1,10)*100
    calc = {}
    data_comp = np.load(str(p1)+'/analysis_collection/results_clustered_app_distribution/results_mean_err.npz')
    data_comp = data_comp['mean']
    data = {}
    for i_n, n in enumerate(networks):
        data[n] = data_comp[i_n]
    calc['a'] = a
    for i_n, n in enumerate(networks):
        calc[n] = {}
        for i_c, c in enumerate(clus):
            calc[n][c] = {}
            O   = np.array([sum([data[n][:,i_c,i] for i in range(5)])]).flatten()/N_sim
            DF  = O/(np.array([sum([data[n][:,i_c,i] for i in [2,3]])]).flatten()/N_sim)
            red = ((O/O[0])-1)*100
            calc[n][c]["absolute"] = {}
            calc[n][c]["reduction"] = {}
            calc[n][c]["absolute"][str(np.round(DF[0]))] = list(O)
            calc[n][c]["reduction"][str(np.round(DF[0]))] = list(red)

    contact_calc = {}
    contacts = np.load(str(p2)+"/results_clustered_ontacts/results_mean_err.npz")
    contacts = contacts['mean']
    contact_calc['a']  = a
    for i_n, n in enumerate(networks):
        contact_calc[n] = {}
        for i_c, c in enumerate(clus):
            c = eval(c)
            contact_calc[n][c] = {}
            contact_calc[n][c]['counts'] = contacts[:,i_c,i_n]

    return calc, contact_calc
def plot_figure():
    networks = ['ER','SW','ER_exp','SW_exp']
    calc, contact_calc = get_data_calculations()
    fig, axs = plt.subplots(2,4,sharey = "row")

    for i_n, n in enumerate(networks):
        axs[1,i_n].set_xlabel('app participation $a$ ' ,loc='right')
        axs[1,i_n].spines['top'].set_visible(False)

        for i_c, c in enumerate(clus):

            axs[0,i_n].plot(calc['a'], calc[n][c]["reduction"]['4.0'], marker = 'o', color = colors[i_c], ls = lss[i_c],\
                            label = labels[i_c])
            axs[0,i_n].yaxis.set_major_formatter(mtick.PercentFormatter())

            c = eval(c)

            for i_r, r in enumerate(res_keys):
                if r == 'counts':
                    axs[1,i_n].plot(contact_calc['a'], contact_calc[n][c][r]/(N_sim*10),\
                    label = labels[i_c], marker = 'o', color = colors[i_c], ls = lss[i_c])

        for i in range(2):
            axs[i,i_n].spines['right'].set_visible(False)
            axs[i,i_n].xaxis.set_major_formatter(mtick.PercentFormatter())
            axs[i,i_n].spines['top'].set_visible(False)

        axs[1,i_n].plot(calc['a'], (calc['a']/100)**2 ,color = 'green', label = r"$a^2$")

    axs[1,0].legend()
    axs[0,0].set_ylabel(r'outbreak size reduction' ,loc='bottom')
    axs[1,0].set_ylabel(r'fraction found contacts' ,loc='bottom')
    plt.show()


def clustered_app_sparse():
    """
    The following function extracts the data for clustered app distribution
    in the network and visualizes it
    """
    data = np.load('results_clustered_app_distribution_sparse_0_1_full/results_mean_err.npz')['mean']
    calc = {}
    for i_n, n in enumerate(networks):
        calc[n] = {}
        for i_c, c in enumerate(clus):
            O   = np.array([sum([data[:,i_n,i_c,i] for i in range(5)])]).flatten()/N_sim
            DF  = O/(np.array([sum([data[:,i_n,i_c,i] for i in [2,3]])]).flatten()/N_sim)
            red = ((O/O[0])-1)*100
            print(str(np.round(DF[0])))
            calc[n][c] = red

    contact_counts = np.load(str(p2)+"/results_clustered_contacts_sparse_0_1_hist/results_mean_err.npz")['mean']

    fig, axs = plt.subplots(1,2)

    axs[0].set_ylabel('% of links with both app-user ' ,loc='bottom')
    axs[1].set_ylabel('outbreak size reduction' ,loc='bottom')
    axs[0].spines['top'].set_visible(False)
    axs[1].spines['bottom'].set_visible(False)

    for i in range(2):
        axs[i].spines['right'].set_visible(False)
        axs[i].set_xticks([])
        axs[i].yaxis.set_major_formatter(mtick.PercentFormatter())

    for i_n, n in enumerate(networks):
        for i_c, c in enumerate(clus):
            if c == 'True':
                pos = 0.2
                alpha = 0.5
                hatch = '\\'
            else:
                pos = -0.2
                alpha = 1
                hatch = ''

            axs[0].bar(i_n+pos, contact_counts[i_c,i_n]/(N_sim*10)*100,color = colors[i_n], width = 0.2,alpha = alpha,hatch = hatch)
            axs[1].bar(i_n+pos, calc[n][c],color = colors[i_n], width = 0.2,alpha = alpha,hatch = hatch)

    plt.show()

if __name__ == '__main__':
    clustered_app_sparse()
    #clustered_app_old_version()
