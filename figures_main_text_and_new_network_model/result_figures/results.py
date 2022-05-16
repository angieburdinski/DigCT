import simplejson as json
import json
import numpy as np
from pathlib import Path
p1 = Path(__file__).parents[2]
p2 = Path(__file__).parents[0]

def convert():

    liste = {"exp":"exp","exp_sw":"sw_exp","sw":"sw","random":"no_lockdown","lockdown":"lockdown"}
    with open(str(p1)+'/analysis_collection/data_main.json','r') as json_file:
        data = json.load(json_file)
    for k, v in data.items():
        if k!= "a":
            v['2.4'] = v['2.0']
            del v['2.0']
            data[k] = [v]

    with open(str(p2)+'/results.json', 'w') as outfile:
        json.dump(data, outfile)

with open('results.json','r') as f:
    results = json.load(f)

a = np.array(results.pop('a'))
i30 = np.argmin(np.abs(a-0.3))
aval = [0, 0.3, 0.5, 0.8]
andx = [np.argmin(np.abs(a-v)) for v in aval]

networks = ['random','exp','sw','exp_sw']

if __name__ == "__main__":
    convert()
