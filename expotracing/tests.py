from extendedmodel import mixed_tracing as mt
from tools import analysis as an
import numpy as np
population_size = 80e6
t = np.linspace(0,150,1000)
model = mt(N = population_size,quarantine_S_contacts = False)

parameters = {
        'R0': 2.5,
        'q': 0.8,
        'app_participation': 0.33,
        'chi':1/2.5,
        'recovery_rate' : 1/8,
        'alpha' : 1/2,
        'beta' : 1/2,
        'number_of_contacts' : 6.3,
        'x':0.4,
        'y':0.1,
        'z':0.64,
        'I_0' : 1000,
        'omega':1/10
        }

q = np.linspace(0,1,6)
app_participation = np.linspace(0,1,50)

results = an(model,parameters,t).two_range_result('app_participation',app_participation,'q',q,['X','I_S'])

#results = an(model,parameters,t).range_result('app_participation',app_participation,['T','X'])
