import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

p1 = Path(__file__).parents[2]
p2 = Path(__file__).parents[0]

network_keys = ['sw','sw_exp']
clus = ['True','False']
res_keys = ['app','counts']
N_sim  = 200_000

def get_data_calculations():
    data = {}
    for i_n, n in enumerate(network_keys):
        data[n] = {}
        data[n] = np.load(str(p1)+'/analysis_collection/tracing_sim/results_'+str(n)\
                            +'_clustered/results_mean_err.npz')
        data[n] = data[n]['mean']
    calc = {}
    calc['a']= np.linspace(0,1,10)*100

    for i_n, n in enumerate(network_keys):
        calc[n] = {}
        for i_c, c in enumerate(clus):
            calc[n][c] = {}
            O   = np.array([sum([data[n][i_c,:,i] for i in range(5)])]).flatten()/N_sim
            DF  = O/(np.array([sum([data[n][i_c,:,i] for i in [2,3]])]).flatten()/N_sim)
            red = ((O/O[0])-1)*100
            calc[n][c]["absolute"] = {}
            calc[n][c]["reduction"] = {}
            calc[n][c]["absolute"][str(np.round(DF[0]))] = list(O)
            calc[n][c]["reduction"][str(np.round(DF[0]))] = list(red)

    contact_calc = {}
    contacts = np.load(str(p2)+'/results_clustered_found_contacts_test_v2/results.npy',allow_pickle=True)

    contact_calc['a']  = np.linspace(0.001,0.999,10)*100
    for i_n, n in enumerate(network_keys):
        contact_calc[n] = {}
        for i_c, c in enumerate(clus):
            c = eval(c)
            contact_calc[n][c] = {}
            for i_r, r in enumerate(res_keys):
                x = contacts[0,:,i_n,i_c]

                if r == 'counts':
                    contact_calc[n][c][r] = x[:,1]

                if r == 'app':
                    contact_calc[n][c][r] = x[:,0]


    return calc, contact_calc



def plot_clustered():
    calc, contact_calc = get_data_calculations()
    #calc = get_data_calculations()
    fig, axs = plt.subplots(2,3,sharey = "col")

    for i_n, n in enumerate(network_keys):
        for i_c, c in enumerate(clus):
            axs[i_n,0].plot(calc['a'], calc[n][c]["reduction"]['4.0'], marker = '.',\
                            label = 'clustered = '+ str(c))
            axs[i_n,0].yaxis.set_major_formatter(mtick.PercentFormatter())


            c = eval(c)
            for i_r, r in enumerate(res_keys):
                if r == 'counts':
                    axs[i_n,1].plot(contact_calc['a'], contact_calc[n][c][r]/(N_sim*10),\
                    label = 'clustered = '+ str(c),marker = ".")
                if r == 'app':
                    print(contact_calc['a'][3])
                    axs[i_n,2].hist(contact_calc[n][c][r][3], bins = np.arange(0,N_sim+N_sim/100,N_sim/100),\
                    label = 'clustered = '+ str(c),alpha = 0.6)

        for i in range(2):
            axs[i_n,i].xaxis.set_major_formatter(mtick.PercentFormatter())
            axs[1,i].set_xlabel('app participation $a$ [%]')
            axs[1,2].set_xlabel('node index')
        for i in range(3):
            #axs[i_n,i].set_title(n)
            axs[i_n,i].legend()
        axs[i_n,0].set_ylabel(r'outbreak size reduction' )
        axs[i_n,1].set_ylabel(r'fraction found contacts' )
        axs[i_n,2].set_ylabel(r'app participants' )

    plt.show()
if __name__ == '__main__':
    plot_clustered()
