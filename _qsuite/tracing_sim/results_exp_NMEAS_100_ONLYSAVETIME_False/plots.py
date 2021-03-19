import pickle
import matplotlib.pyplot as plt
hfont = {'fontname':'Helvetica'}
plt.rcParams.update({'font.size': 13})
import numpy as np
from matplotlib.lines import Line2D
import gzip
import qsuite_config as cf
data = pickle.load(gzip.open('/Users/angeliqueburdinski/Desktop/Arbeit/tracing_sim/results_exp_NMEAS_100_ONLYSAVETIME_False/results.p.gz','rb'))
data = np.array(data)
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
print(data.shape)
def plot_func():
    positions = (0, 5, 10, 15, 20, 25)
    labels = ('0',"0.2","0.4", "0.6", "0.8",'1.0')
    plt.xticks(positions, labels)
    lines = [Line2D([0], [0], color = colors[x], linewidth=3, linestyle='dotted') for x in range(len(cf.q))]
    labels = ['q = 0','q = 0.1','q = 0.3','q = 0.5','q = 0.7','q = 0.9']
    plt.legend(lines, labels)
    plt.xlabel('a',**hfont)

def infectious():
    I00 = []
    for x in range(len(cf.q)):
        I00.append(data[:,0,0,:,x,0] + data[:,0,0,:,x,1])
    I00 = np.array(I00)
    mean = np.mean(I00,axis = 1)
    standard_dev = np.std(I00,axis = 1)
    for x in range(6):
        plt.plot(mean[x], alpha = 1, color = colors[x])
        plt.fill_between(range(25),mean[x]-standard_dev[x], mean[x]+standard_dev[x], alpha = 0.5,color = colors[x])
    plot_func()
    plt.ylim(0,40_000)
    plt.ylabel(' $I_{S_{max}}$',**hfont)
    plt.title('R0 = 2.5 z = 0.32')
    plt.savefig('exp R0 = 2_5 z = 0_32')
    plt.show()

    I01 = []
    for x in range(len(cf.q)):
        I01.append(data[:,0,1,:,x,0] + data[:,0,1,:,x,1])
    I01 = np.array(I01)
    mean = np.mean(I01,axis = 1)
    standard_dev = np.std(I01,axis = 1)
    for x in range(6):
        plt.plot(mean[x], alpha = 1, color = colors[x])
        plt.fill_between(range(25),mean[x]-standard_dev[x], mean[x]+standard_dev[x], alpha = 0.5,color = colors[x])
    plot_func()
    plt.ylim(0,40_000)
    plt.ylabel(' $I_{S_{max}}$',**hfont)
    plt.title('R0 = 2.5 z = 0.64')
    plt.savefig('exp R0 = 2_5 z = 0_64')
    plt.show()

    I10 = []
    for x in range(len(cf.q)):
        I10.append(data[:,1,0,:,x,0] + data[:,1,0,:,x,1])
    I10 = np.array(I10)
    mean = np.mean(I10,axis = 1)
    standard_dev = np.std(I10,axis = 1)
    for x in range(6):
        plt.plot(mean[x], alpha = 1, color = colors[x])
        plt.fill_between(range(25),mean[x]-standard_dev[x], mean[x]+standard_dev[x], alpha = 0.5,color = colors[x])
    plot_func()
    plt.ylim(0,40_000)
    plt.ylabel(' $I_{S_{max}}$',**hfont)
    plt.title('R0 = 2.6 z = 0.32')
    plt.savefig('exp R0 = 2_6 z = 0_32')
    plt.show()

    I11 = []
    for x in range(len(cf.q)):
        I11.append(data[:,1,1,:,x,0] + data[:,1,1,:,x,1])
    I11 = np.array(I11)
    mean = np.mean(I11,axis = 1)
    standard_dev = np.std(I11,axis = 1)
    for x in range(6):
        plt.plot(mean[x], alpha = 1, color = colors[x])
        plt.fill_between(range(25),mean[x]-standard_dev[x], mean[x]+standard_dev[x], alpha = 0.5,color = colors[x])
    plot_func()
    plt.ylim(0,40_000)
    plt.ylabel(' $I_{S_{max}}$',**hfont)
    plt.title('R0 = 2.6 z = 0.64')
    plt.savefig('exp R0 = 2_6 z = 0_64')
    plt.show()
#infectious()

def removed():
    I00 = []
    for x in range(len(cf.q)):
        I00.append(data[:,0,0,:,x,2] + data[:,0,0,:,x,3])
    I00 = np.array(I00)
    mean = np.mean(I00,axis = 1)
    standard_dev = np.std(I00,axis = 1)
    for x in range(6):
        plt.plot(mean[x], alpha = 1, color = colors[x])
        plt.fill_between(range(25),mean[x]-standard_dev[x], mean[x]+standard_dev[x], alpha = 0.5,color = colors[x])
    plot_func()
    plt.ylim(0,200_000)
    plt.ylabel(' $R_{max}$',**hfont)
    plt.title('R0 = 2.5 z = 0.32')
    plt.savefig('exp R0 = 2_5 z = 0_32 Rmax')
    plt.show()

    I01 = []
    for x in range(len(cf.q)):
        I01.append(data[:,0,1,:,x,2] + data[:,0,1,:,x,3])
    I01 = np.array(I01)
    mean = np.mean(I01,axis = 1)
    standard_dev = np.std(I01,axis = 1)
    for x in range(6):
        plt.plot(mean[x], alpha = 1, color = colors[x])
        plt.fill_between(range(25),mean[x]-standard_dev[x], mean[x]+standard_dev[x], alpha = 0.5,color = colors[x])
    plot_func()
    plt.ylim(0,200_000)
    plt.ylabel(' $R_{max}$',**hfont)
    plt.title('R0 = 2.5 z = 0.64')
    plt.savefig('exp R0 = 2_5 z = 0_64 Rmax')
    plt.show()

    I10 = []
    for x in range(len(cf.q)):
        I10.append(data[:,1,0,:,x,2] + data[:,1,0,:,x,3])
    I10 = np.array(I10)
    mean = np.mean(I10,axis = 1)
    standard_dev = np.std(I10,axis = 1)
    for x in range(6):
        plt.plot(mean[x], alpha = 1, color = colors[x])
        plt.fill_between(range(25),mean[x]-standard_dev[x], mean[x]+standard_dev[x], alpha = 0.5,color = colors[x])
    plot_func()
    plt.ylim(0,200_000)
    plt.ylabel(' $R_{max}$',**hfont)
    plt.title('R0 = 2.6 z = 0.32')
    plt.savefig('exp R0 = 2_6 z = 0_32 Rmax')
    plt.show()

    I11 = []
    for x in range(len(cf.q)):
        I11.append(data[:,1,1,:,x,2] + data[:,1,1,:,x,3])
    I11 = np.array(I11)
    mean = np.mean(I11,axis = 1)
    standard_dev = np.std(I11,axis = 1)
    for x in range(6):
        plt.plot(mean[x], alpha = 1, color = colors[x])
        plt.fill_between(range(25),mean[x]-standard_dev[x], mean[x]+standard_dev[x], alpha = 0.5,color = colors[x])
    plot_func()
    plt.ylim(0,200_000)
    plt.ylabel(' $R_{max}$',**hfont)
    plt.title('R0 = 2.6 z = 0.64')
    plt.savefig('exp R0 = 2_6 z = 0_64 Rmax')
    plt.show()
#removed()
def known_quarantined():
    I00 = []
    for x in range(len(cf.q)):
        I00.append(data[:,0,0,:,x,4] + data[:,0,0,:,x,5])
    I00 = np.array(I00)
    mean = np.mean(I00,axis = 1)
    standard_dev = np.std(I00,axis = 1)
    for x in range(6):
        plt.plot(mean[x], alpha = 1, color = colors[x])
        plt.fill_between(range(25),mean[x]-standard_dev[x], mean[x]+standard_dev[x], alpha = 0.5,color = colors[x])
    plot_func()
    plt.ylim(0,90_000)
    plt.ylabel(' $X_{max}$',**hfont)
    plt.title('R0 = 2.5 z = 0.32')
    plt.savefig('exp R0 = 2_5 z = 0_32 Xmax')
    plt.show()

    I01 = []
    for x in range(len(cf.q)):
        I01.append(data[:,0,1,:,x,4] + data[:,0,1,:,x,5])
    I01 = np.array(I01)
    mean = np.mean(I01,axis = 1)
    standard_dev = np.std(I01,axis = 1)
    for x in range(6):
        plt.plot(mean[x], alpha = 1, color = colors[x])
        plt.fill_between(range(25),mean[x]-standard_dev[x], mean[x]+standard_dev[x], alpha = 0.5,color = colors[x])
    plot_func()
    plt.ylim(0,90_000)
    plt.ylabel(' $X_{max}$',**hfont)
    plt.title('R0 = 2.5 z = 0.64')
    plt.savefig('exp R0 = 2_5 z = 0_64 Xmax')
    plt.show()

    I10 = []
    for x in range(len(cf.q)):
        I10.append(data[:,1,0,:,x,4] + data[:,1,0,:,x,5])
    I10 = np.array(I10)
    mean = np.mean(I10,axis = 1)
    standard_dev = np.std(I10,axis = 1)
    for x in range(6):
        plt.plot(mean[x], alpha = 1, color = colors[x])
        plt.fill_between(range(25),mean[x]-standard_dev[x], mean[x]+standard_dev[x], alpha = 0.5,color = colors[x])
    plot_func()
    plt.ylim(0,90_000)
    plt.ylabel(' $X_{max}$',**hfont)
    plt.title('R0 = 2.6 z = 0.32')
    plt.savefig('exp R0 = 2_6 z = 0_32 Xmax')
    plt.show()

    I11 = []
    for x in range(len(cf.q)):
        I11.append(data[:,1,1,:,x,4] + data[:,1,1,:,x,5])
    I11 = np.array(I11)
    mean = np.mean(I11,axis = 1)
    standard_dev = np.std(I11,axis = 1)
    for x in range(6):
        plt.plot(mean[x], alpha = 1, color = colors[x])
        plt.fill_between(range(25),mean[x]-standard_dev[x], mean[x]+standard_dev[x], alpha = 0.5,color = colors[x])
    plot_func()
    plt.ylim(0,90_000)
    plt.ylabel(' $X_{max}$',**hfont)
    plt.title('R0 = 2.6 z = 0.64')
    plt.savefig('exp R0 = 2_6 z = 0_64 Xmax')
    plt.show()
#known_quarantined()

def unknown_quarantined():
    I00 = []
    for x in range(len(cf.q)):
        I00.append(data[:,0,0,:,x,6])
    I00 = np.array(I00)
    mean = np.mean(I00,axis = 1)
    standard_dev = np.std(I00,axis = 1)
    for x in range(6):
        plt.plot(mean[x], alpha = 1, color = colors[x])
        plt.fill_between(range(25),mean[x]-standard_dev[x], mean[x]+standard_dev[x], alpha = 0.5,color = colors[x])
    plot_func()
    plt.ylim(0,30_000)
    plt.ylabel(' $C_{max}$',**hfont)
    plt.title('R0 = 2.5 z = 0.32')
    plt.savefig('exp R0 = 2_5 z = 0_32 Cmax')
    plt.show()

    I01 = []
    for x in range(len(cf.q)):
        I01.append(data[:,0,1,:,x,6])
    I01 = np.array(I01)
    mean = np.mean(I01,axis = 1)
    standard_dev = np.std(I01,axis = 1)
    for x in range(6):
        plt.plot(mean[x], alpha = 1, color = colors[x])
        plt.fill_between(range(25),mean[x]-standard_dev[x], mean[x]+standard_dev[x], alpha = 0.5,color = colors[x])
    plot_func()
    plt.ylim(0,30_000)
    plt.ylabel(' $C_{max}$',**hfont)
    plt.title('R0 = 2.5 z = 0.64')
    plt.savefig('exp R0 = 2_5 z = 0_64 Cmax')
    plt.show()

    I10 = []
    for x in range(len(cf.q)):
        I10.append(data[:,1,0,:,x,6])
    I10 = np.array(I10)
    mean = np.mean(I10,axis = 1)
    standard_dev = np.std(I10,axis = 1)
    for x in range(6):
        plt.plot(mean[x], alpha = 1, color = colors[x])
        plt.fill_between(range(25),mean[x]-standard_dev[x], mean[x]+standard_dev[x], alpha = 0.5,color = colors[x])
    plot_func()
    plt.ylim(0,30_000)
    plt.ylabel(' $C_{max}$',**hfont)
    plt.title('R0 = 2.6 z = 0.32')
    plt.savefig('exp R0 = 2_6 z = 0_32 Cmax')
    plt.show()

    I11 = []
    for x in range(len(cf.q)):
        I11.append(data[:,1,1,:,x,6])
    I11 = np.array(I11)
    mean = np.mean(I11,axis = 1)
    standard_dev = np.std(I11,axis = 1)
    for x in range(6):
        plt.plot(mean[x], alpha = 1, color = colors[x])
        plt.fill_between(range(25),mean[x]-standard_dev[x], mean[x]+standard_dev[x], alpha = 0.5,color = colors[x])
    plot_func()
    plt.ylim(0,30_000)
    plt.ylabel(' $C_{max}$',**hfont)
    plt.title('R0 = 2.6 z = 0.64')
    plt.savefig('exp R0 = 2_6 z = 0_64 Cmax')
    plt.show()
#unknown_quarantined()
