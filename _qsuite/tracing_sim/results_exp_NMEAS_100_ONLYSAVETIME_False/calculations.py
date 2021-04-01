import pickle
import matplotlib.pyplot as plt
hfont = {'fontname':'Helvetica'}
plt.rcParams.update({'font.size': 13})
import numpy as np
from matplotlib.lines import Line2D
import gzip
import qsuite_config as cf
data = pickle.load(gzip.open('/Users/angeliqueburdinski/expotracing/_qsuite/tracing_sim/results_exp_NMEAS_100_ONLYSAVETIME_False/results.p.gz','rb'))
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
def QvsR():
    R00 = []
    Q00 = []
    for x in range(len(cf.q)):
        R00.append(data[:,0,0,:,x,2] + data[:,0,0,:,x,3])
        Q00.append(data[:,0,0,:,x,4] + data[:,0,0,:,x,5] + data[:,0,0,:,x,6])

    R00 = np.array(R00)
    Q00 = np.array(Q00)
    Diff = Q00 / R00
    meanDiff = np.mean(Diff,axis = 1)
    standard_devDiff = np.std(Diff,axis = 1)

    for x in range(6):
        plt.plot(meanDiff[x], alpha = 1, color = colors[x])
        plt.fill_between(range(25),meanDiff[x]-standard_devDiff[x], meanDiff[x]+standard_devDiff[x], alpha = 0.5,color = colors[x])
    positions = (0, 5, 10, 15, 20, 25)
    labels = ('0',"0.2","0.4", "0.6", "0.8",'1.0')
    plt.xticks(positions, labels)
    lines = [Line2D([0], [0], color = colors[x], linewidth=3, linestyle='dotted') for x in range(len(cf.q))]
    labels = ['q = 0','q = 0.1','q = 0.3','q = 0.5','q = 0.7','q = 0.9']
    plt.legend(lines, labels)
    plt.title('R0 = 2.5 z = 0.32')
    plt.ylabel('quarantined vs removed')
    plt.xlabel('a',**hfont)
    plt.savefig('QvsR R0 = 2_5 z = 0_32')
    plt.show()

    R01 = []
    Q01 = []
    for x in range(len(cf.q)):
        R01.append(data[:,0,1,:,x,2] + data[:,0,1,:,x,3])
        Q01.append(data[:,0,1,:,x,4] + data[:,0,1,:,x,5] + data[:,0,1,:,x,6])

    R01 = np.array(R01)
    Q01 = np.array(Q01)
    Diff = Q01 / R01
    meanDiff = np.mean(Diff,axis = 1)
    standard_devDiff = np.std(Diff,axis = 1)

    for x in range(6):
        plt.plot(meanDiff[x], alpha = 1, color = colors[x])
        plt.fill_between(range(25),meanDiff[x]-standard_devDiff[x], meanDiff[x]+standard_devDiff[x], alpha = 0.5,color = colors[x])
    positions = (0, 5, 10, 15, 20, 25)
    labels = ('0',"0.2","0.4", "0.6", "0.8",'1.0')
    plt.xticks(positions, labels)
    lines = [Line2D([0], [0], color = colors[x], linewidth=3, linestyle='dotted') for x in range(len(cf.q))]
    labels = ['q = 0','q = 0.1','q = 0.3','q = 0.5','q = 0.7','q = 0.9']
    plt.legend(lines, labels)
    plt.title('R0 = 2.5 z = 0.64')
    plt.ylabel('quarantined vs removed')
    plt.xlabel('a',**hfont)
    plt.savefig('QvsR R0 = 2_5 z = 0_64')
    plt.show()


    R10 = []
    Q10 = []
    for x in range(len(cf.q)):
        R10.append(data[:,1,0,:,x,2] + data[:,1,0,:,x,3])
        Q10.append(data[:,1,0,:,x,4] + data[:,1,0,:,x,5] + data[:,1,0,:,x,6])

    R10 = np.array(R10)
    Q10 = np.array(Q10)
    Diff = Q10 / R10
    meanDiff = np.mean(Diff,axis = 1)
    standard_devDiff = np.std(Diff,axis = 1)

    for x in range(6):
        plt.plot(meanDiff[x], alpha = 1, color = colors[x])
        plt.fill_between(range(25),meanDiff[x]-standard_devDiff[x], meanDiff[x]+standard_devDiff[x], alpha = 0.5,color = colors[x])
    positions = (0, 5, 10, 15, 20, 25)
    labels = ('0',"0.2","0.4", "0.6", "0.8",'1.0')
    plt.xticks(positions, labels)
    lines = [Line2D([0], [0], color = colors[x], linewidth=3, linestyle='dotted') for x in range(len(cf.q))]
    labels = ['q = 0','q = 0.1','q = 0.3','q = 0.5','q = 0.7','q = 0.9']
    plt.legend(lines, labels)
    plt.title('R0 = 2.6 z = 0.32')
    plt.ylabel('quarantined vs removed')
    plt.xlabel('a',**hfont)
    plt.savefig('QvsR R0 = 2_6 z = 0_32')
    plt.show()

    R11 = []
    Q11 = []
    for x in range(len(cf.q)):
        R11.append(data[:,1,1,:,x,2] + data[:,1,1,:,x,3])
        Q11.append(data[:,1,1,:,x,4] + data[:,1,1,:,x,5] + data[:,1,1,:,x,6])

    R11 = np.array(R11)
    Q11 = np.array(Q11)
    Diff = Q11 / R11
    meanDiff = np.mean(Diff,axis = 1)
    standard_devDiff = np.std(Diff,axis = 1)

    for x in range(6):
        plt.plot(meanDiff[x], alpha = 1, color = colors[x])
        plt.fill_between(range(25),meanDiff[x]-standard_devDiff[x], meanDiff[x]+standard_devDiff[x], alpha = 0.5,color = colors[x])
    positions = (0, 5, 10, 15, 20, 25)
    labels = ('0',"0.2","0.4", "0.6", "0.8",'1.0')
    plt.xticks(positions, labels)
    lines = [Line2D([0], [0], color = colors[x], linewidth=3, linestyle='dotted') for x in range(len(cf.q))]
    labels = ['q = 0','q = 0.1','q = 0.3','q = 0.5','q = 0.7','q = 0.9']
    plt.legend(lines, labels)
    plt.title('R0 = 2.6 z = 0.64')
    plt.ylabel('quarantined vs removed')
    plt.xlabel('a',**hfont)
    plt.savefig('QvsR R0 = 2_6 z = 0_64')
    plt.show()

def darkfactor():
    R11 = []
    X11 = []
    Q11 = []
    for x in range(len(cf.q)):
        R11.append(data[:,0,1,:,x,2] + data[:,0,1,:,x,3])
        Q11.append(data[:,0,1,:,x,6])
        X11.append(data[:,0,1,:,x,4] + data[:,0,1,:,x,5])
    R11 = np.array(R11)
    X11 = np.array(X11)
    Q11 = np.array(Q11)
    DF = (R11+X11+Q11)/(X11+Q11)
    meanDF = np.mean(DF,axis = 1)
    standard_devDF = np.std(DF,axis = 1)
    for x in range(6):
        plt.plot(meanDF[x], alpha = 1, color = colors[x])
        plt.fill_between(range(25),meanDF[x]-standard_devDF[x], meanDF[x]+standard_devDF[x], alpha = 0.5,color = colors[x])
    positions = (0, 6, 12, 18, 24)
    labels = ('0',"0.25","0.5", "0.75",'1.0')
    plt.xticks(positions, labels)
    lines = [Line2D([0], [0], color = colors[x], linewidth=3, linestyle='-') for x in [1,2,3,4,5]]
    labels = ['q = 0.1','q = 0.3','q = 0.5','q = 0.7','q = 0.9']
    #plt.legend(lines, labels)
    plt.ylabel(r'$DF$')
    plt.xlabel(r'$a$',**hfont)
    plt.savefig('exp DF R0 = 2_5 z = 0_64')
    plt.show()
def RX():
    RX01 = []
    RX00 = []
    for x in range(len(cf.q)):
        RX01.append((data[:,0,1,:,x,2] + data[:,0,1,:,x,3]+data[:,0,1,:,x,4] + data[:,0,1,:,x,5]+data[:,0,1,:,x,6])/200_000)
        RX00.append((data[:,0,0,:,x,2] + data[:,0,0,:,x,3]+data[:,0,0,:,x,4] + data[:,0,0,:,x,5]+data[:,0,0,:,x,6])/200_000)
    meanRX01 = np.mean(RX01,axis = 1)
    standard_devRX01 = np.std(RX01,axis = 1)
    meanRX00 = np.mean(RX00,axis = 1)
    standard_devRX00 = np.std(RX00,axis = 1)
    for x in range(6):
        plt.plot(meanRX01[x], alpha = 1, color = colors[x])
        plt.plot(meanRX00[x], alpha = 0.7, color = colors[x])
        plt.fill_between(range(25),meanRX01[x]-standard_devRX01[x], meanRX01[x]+standard_devRX01[x], alpha = 0.7,color = colors[x])
        plt.fill_between(range(25),meanRX00[x]-standard_devRX00[x], meanRX00[x]+standard_devRX00[x], alpha = 0.3,color = colors[x])
    positions = (0, 6, 12, 18, 24)
    labels = ('0',"0.25","0.5", "0.75",'1.0')
    plt.xticks(positions, labels)
    lines = [Line2D([0], [0], color = colors[x], linewidth=3, linestyle='-') for x in range(len(cf.q))]
    labels = ['q = 0','q = 0.1','q = 0.3','q = 0.5','q = 0.7','q = 0.9']
    plt.legend(lines, labels)
    plt.ylabel(r'$R(t \rightarrow \infty)+X(t \rightarrow \infty)$')
    plt.ylim(0,0.8)
    plt.xlabel(r'$a$',**hfont)
    plt.savefig('expRX R0 = 2_5',dpi = 300)
    plt.show()
dcolors = [
        'black',
        'brown',
        'steelblue',
        'maroon',
        'darkslategrey',
        'saddlebrown',
        'olive',
        'midnightblue',
        'darkmagenta',
        'darkolivegreen',
        'firebrick'
        ]
def RX_new():
    RX01 = []
    #RX00 = []
    for x in range(len(cf.q)):
        RX01.append(((data[:,0,1,:,x,2] + data[:,0,1,:,x,3]+data[:,0,1,:,x,4] + data[:,0,1,:,x,5]+data[:,0,1,:,0,6])/200_000))
        #RX00.append((data[:,0,0,:,x,2] + data[:,0,0,:,x,3]+data[:,0,0,:,x,4] + data[:,0,0,:,x,5]+data[:,0,0,:,x,6])/200_000)
    meanRX01 = np.mean(RX01,axis = 1)
    #tandard_devRX01 = np.std(RX01,axis = 1)
    #meanRX00 = np.mean(RX00,axis = 1)
    #standard_devRX00 = np.std(RX00,axis = 1)
    for x in range(6):
        plt.plot((meanRX01[x]/meanRX01[x][0]-1)*100, alpha = 1, color = dcolors[x])
        #plt.plot(meanRX00[x], alpha = 0.7, color = lcolors[x])
        #plt.fill_between(range(25),(meanRX01[x]/meanRX01[x][0]-1)*100-standard_devRX01[x], (meanRX01[x]/meanRX01[x][0]-1)*100+standard_devRX01[x], alpha = 0.7,color = dcolors[x])
        #plt.fill_between(range(25),meanRX00[x]-standard_devRX00[x], meanRX00[x]+standard_devRX00[x], alpha = 0.3,color = lcolors[x])
    positions = (0, 6, 12, 18, 24)
    labels = ('0',"0.25","0.5", "0.75",'1.0')
    plt.xticks(positions, labels)
    lines = [Line2D([0], [0], color = dcolors[x], linewidth=3, linestyle='-') for x in range(len(cf.q))]
    labels = ['q = 0','q = 0.1','q = 0.3','q = 0.5','q = 0.7','q = 0.9']
    plt.legend(lines, labels)
    plt.ylabel(r'reduction of $RX_{max}$ to $RX_{max}(a = 0)$')
    plt.ylim(-85,0)
    #plt.vlines(8,meanRX01[2][8],meanRX01[3][8])
    plt.xlabel(r'$a$',**hfont)
    plt.savefig('expredRX',dpi = 300)
    plt.show()
RX_new()
