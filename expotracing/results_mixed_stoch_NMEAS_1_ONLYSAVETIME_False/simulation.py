from extendedmodel import stoch_mixed_tracing
from tools import (analysis,configuration_network)
import numpy as np

def simulation_code(kwargs):

    def sim(N, k0, time, parameter):
        q = [0,0.2,0.4,0.6,0.8]
        a = np.linspace(0,0.8,50)
        G = configuration_network(N,k0).build()
        model = stoch_mixed_tracing(G,quarantine_S_contacts = True)
        t,result = analysis(model,parameter).stoch_two_range_result('app_participation',a,'q',q,time)


    t, result = sim(**kwargs)

    return result
