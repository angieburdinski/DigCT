import pickle
import qsuite_config as cf
import numpy as np

S, E, I, R, X = list("SEIRX")
Sa, Ea, Ia, Ra, Xa = [letter+"a" for letter in "SEIRX"]
Za = "Za"
Ya = "Ya"

compartments = [S,E,I,R,X,Sa,Ea,Ia,Ra,Xa,Ya,Za]

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

def flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, str) and not isinstance(el, dict):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result

def sumres(result,compartments):
    return sum([result[C] for C in compartments])

with open('results.p','rb') as f:
    data = pickle.load(f)

maxlen = max([ len(res['S']) for res in flatten(data) ])

means = []
stds = []

for iphase, phase in enumerate(cf.phases):
    means.append([])
    stds.append([])
    for ia, a in enumerate(cf.a_s):
        result_mean = {}
        result_std = {}
        for C in compartments:
            vals = [ make_length(data[meas][iphase][ia][C],maxlen) for meas in cf.measurements ]
            thismean = np.mean(vals,axis=0)
            thisstd = np.std(vals,axis=0)
            result_mean[C] = thismean
            result_std[C] = thisstd
        for Cname, C in [
                    ('Stot', ['S','Sa']),
                    ('Itot', ['I','Ia']),
                    ('Etot', ['E','Ea']),
                ]:
            vals = [ make_length(sumres(data[meas][iphase][ia],C),maxlen) for meas in cf.measurements ]
            thismean = np.mean(vals,axis=0)
            thisstd = np.std(vals,axis=0)
            result_mean[Cname] = thismean
            result_std[Cname] = thisstd

        means[-1].append(result_mean)
        stds[-1].append(result_std)


with open('results_mean_std.p','wb') as f:
    pickle.dump({'means':means,'stds':stds},f)




