import numpy as np

from simulation import simulation_code
from tqdm import tqdm

def make_length(arr,maxlen):
    dL = maxlen - len(arr)
    if dL > 0:
        newa = np.concatenate((arr, np.ones(dL)*arr[-1]))
    else:
        newa = arr

    return newa

def make_equal_length(arr_list):
    maxlen = max([len(a) for a in arr_list])
    new_arr_list = []
    for a in arr_list:
        newa = make_length(a,maxlen)
        new_arr_list.append(newa)
    return new_arr_list

np.random.seed(981736)

N = 10_000
n_meas = 100

kwargs = dict(
    N = N,
    q = 0.3,
    a = 0.3,
    R0 = 2.5,
    quarantiningS = True,
    parameter = {
            'chi':1/2.5,
            'recovery_rate' : 1/7,
            'alpha' : 1/3,
            'beta' : 1/2,
            'number_of_contacts' : 20,
            'x':0.17,
            'I_0' : N*0.01,
            'omega':1/10,
            "y" : 0.1,
            "z": 0.64,
            "R0": 2.5,
            "network_model":'er_network',
            },
    sampling_dt = 1,
    time = 1e7,
    )

import matplotlib.pyplot as pl



results_tracing = []
results_no_trac = []

for meas in tqdm(range(n_meas)):
    kwargs['a'] = 0.3
    t0, result0 = simulation_code(kwargs)
    kwargs['a'] = 0.0
    t1, result1 = simulation_code(kwargs)

    results_tracing.append(result0)
    results_no_trac.append(result1)

results_tracing = np.array(make_equal_length(results_tracing))
results_no_trac = np.array(make_equal_length(results_no_trac))

t0 = np.arange(np.shape(results_tracing)[1])
t1 = np.arange(np.shape(results_no_trac)[1])

mn0 = np.mean(results_tracing,axis=0)
mn1 = np.mean(results_no_trac,axis=0)

err0 = np.std(results_tracing,axis=0)
err1 = np.std(results_no_trac,axis=0)

err0low, md0, err0high = np.percentile(results_tracing,[25,50,75],axis=0)
err1low, md1, err1high = np.percentile(results_no_trac,[25,50,75],axis=0)

pl.plot(t0, md0, label='with tracing (a=0.3)')
pl.plot(t1, md1, label='without tracing')
pl.fill_between(t0, err0low, err0high, alpha=0.2)
pl.fill_between(t1, err1low, err1high, alpha=0.2)
pl.xlabel('time [d]')
pl.ylabel('prevalence')

pl.legend()

pl.gcf().savefig('example.png',dpi=300)
pl.show()
