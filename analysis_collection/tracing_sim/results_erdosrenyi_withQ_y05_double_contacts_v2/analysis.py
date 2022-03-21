import numpy as np
import matplotlib.pyplot as plt
import json
data = np.load('results_mean_err.npz')['mean']

with open('/Users/angeliqueburdinski/research/infectious_diseases/COVID19/tracing/DigCT/analysis_collection/data_new.json') as json_file:
    data_ = json.load(json_file)

data_dict = {}
data_dict["O"] = np.array([sum([data[:,x,0,i] for i in range(5)]) for x in range(4)])/200_000
data_dict["DF"] = (data_dict["O"])/(np.array([sum([data[:,x,0,i] for i in [2,3]]) for x in range(4)])/200_000)
data_dict["red"] =  [(((data_dict["O"][x]/data_dict["O"][x][0])-1)*100) for x in range(4)]
data_dict["absolute"] = {}
data_dict["reduction"] = {}
for i in range(4):
    data_dict["absolute"][str(np.round(data_dict["DF"][i][0]))] = list(data_dict["O"][i])
    data_dict["reduction"][str(np.round(data_dict["DF"][i][0]))] = list(data_dict["red"][i])
fig,ax = plt.subplots(1,2,sharex =True,sharey = True)
for i in ["2.0","4.0","12.0"]:
    ax[0].plot(np.linspace(0,1,25), data_dict["absolute"][i])
    ax[0].set_title("double contacts")
    ax[0].set_ylabel("absolute outbreaksize")
    ax[0].set_xlabel("app")
    ax[1].set_xlabel("app")
    ax[1].plot(np.linspace(0,1,25), data_["no_lockdown"]["absolute"][i])
plt.show()
fig,ax = plt.subplots(1,2,sharex =True,sharey = True)
for i in ["2.0","4.0","12.0"]:
    ax[0].plot(np.linspace(0,1,25), data_dict["reduction"][i])
    ax[0].set_title("double contacts")
    ax[0].set_ylabel("relative outbreaksize")
    ax[0].set_xlabel("app")
    ax[1].set_xlabel("app")
    ax[1].plot(np.linspace(0,1,25), data_["no_lockdown"]["reduction"][i])
plt.show()
