import json
import numpy as np
def old_json_for_ben():

    liste = {"exp":"exp","exp_sw":"sw_exp","sw":"sw","random":"no_lockdown","lockdown":"lockdown"}

    data = {}
    data["a"] = list(np.linspace(0,1,25))

    with open('results_new_run.json','r') as json_file:
        data_new = json.load(json_file)

    for k, v in liste.items():
        data[k] = data_new[v]["absolute"]

    for k, v in data.items():
        if k!= "a":
            v['2.4']  = v.pop('2.0')
            data[k] = [v]

    with open('results.json', 'w') as outfile:
        json.dump(data, outfile)

if __name__ == "__main__":
    old_json_for_ben()
