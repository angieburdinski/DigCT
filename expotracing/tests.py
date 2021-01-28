from extendedmodel import mixed_tracing as mt
from tools import analysis as an
import numpy as np
population_size = 80e6
t = np.linspace(0,365,1000)
model = mt(N = population_size,quarantine_S_contacts = False)

parameters = {
        'R0': 2.5,
        'q': 0.5,
        'app_participation': 0.33,
        'chi':1/2.5,
        'recovery_rate' : 1/6,
        'alpha' : 1/2,
        'beta' : 1/2,
        'number_of_contacts' : 6.3*1/0.33,
        'x':0.4,
        'y':0.1,
        'z':0.64,
        'I_0' : 1000,
        'omega':1/10
        }

q = [0,0.01,0.2,0.4,0.6,0.8]
number_of_contacts = np.linspace(0,100,10)
#a = np.linspace(0,0.8,25)

results = an(model,parameters,t).two_range_result('number_of_contacts',number_of_contacts,'q',q,['R','X'])
