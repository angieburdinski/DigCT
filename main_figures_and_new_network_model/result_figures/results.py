import numpy as np

import simplejson as json

with open('results.json','r') as f:
    results = json.load(f)

a = np.array(results.pop('a'))
i30 = np.argmin(np.abs(a-0.3))
aval = [0, 0.3, 0.5, 0.8]
andx = [np.argmin(np.abs(a-v)) for v in aval]

networks = ['random','exp','sw','exp_sw']
