import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import gzip
from matplotlib.lines import Line2D
import matplotlib.ticker as mtick
import qsuite_config as cf
data =  np.array(pickle.load(gzip.open('results.p.gz','rb')))
data_app_list = data[:,:,:,0]
data = data[:,:,:,1:]
data = np.mean(data, axis = 0)
data_name = ["True","False"]
data_dict = {}
for k in data_name:
    data_dict["applist_"+str(k)] = data_app_list[0,data_name.index(k),5]
    data_dict["O_"+str(k)] = np.array(sum([data[data_name.index(k),:,i] for i in range(5)]))/cf.N
    data_dict["DF_"+str(k)] = (data_dict["O_"+str(k)])/(np.array(sum([data[data_name.index(k),:,i] for i in [2,3]]))/cf.N)
    data_dict["red_"+str(k)] =  (((data_dict["O_"+str(k)]/data_dict["O_"+str(k)][0])-1)*100)

    data_dict["absolute_"+str(k)] = {}
    data_dict["reduction_"+str(k)] = {}
    data_dict["absolute_"+str(k)][str(np.round(data_dict["DF_"+str(k)][0]))] = list(data_dict["O_"+str(k)])
    data_dict["reduction_"+str(k)][str(np.round(data_dict["DF_"+str(k)][0]))] = list(data_dict["red_"+str(k)])
colors = [
    'dimgrey',
    'lightcoral',
]
fig,ax  = plt.subplots()
ax.plot(np.linspace(0,1,10), data_dict["reduction_True"]["4.0"], color = colors[0])
ax.plot(np.linspace(0,1,10), data_dict["reduction_False"]["4.0"], color = colors[1])
ax.set_ylabel("outbreak size reduction")
ax.set_xlabel("app participation")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(100,decimals=0))
axin0 = ax.inset_axes([0.15, 0.1,0.35, 0.35])
axin0.plot(np.linspace(0,1,10), data_dict["absolute_True"]["4.0"], color = colors[0])
axin0.plot(np.linspace(0,1,10), data_dict["absolute_False"]["4.0"], color = colors[1])
axin0.set_ylabel(r'outbreak size $\langle\Omega\rangle$/N')
lines = [Line2D([0], [0], color = colors[0], alpha = 1, linewidth=1, linestyle='-'),
         Line2D([0], [0], color = colors[1], alpha = 1, linewidth=1, linestyle='-')]
labels = ['clustered', 'randomly distributed ']
ax.legend(lines,labels)
plt.show()
