import json
import numpy as np
def old_json_for_ben():

    liste = {"exp":"exp","exp_sw":"sw_exp","sw":"sw","random":"no_lockdown","lockdown":"lockdown"}

    #dat
    #data["a"] = list(np.linspace(0,1,25))
    with open('/Users/angeliqueburdinski/research/infectious_diseases/COVID19/tracing/DigCT/analysis_collection/data_main.json','r') as json_file:
    #with open('results_new_run.json','r') as json_file:
        #data_new = json.load(json_file)
        data = json.load(json_file)
    #for k, v in liste.items():
    #    data[k] = data_new[v+"0.5"]["absolute"]

    for k, v in data.items():
        if k!= "a":

            v['2.4'] = v['2.0']
            del v['2.0']
            #v['2.4'] = v.pop('2.0')
            data[k] = [v]

    with open('results.json', 'w') as outfile:
        json.dump(data, outfile)

if __name__ == "__main__":
    old_json_for_ben()
