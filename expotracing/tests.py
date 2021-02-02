from extendedmodel import mixed_tracing as mt
from tools import analysis as an
from plots import plot
import matplotlib.pyplot as plt
import numpy as np

population_size = 80e6
t = np.linspace(0,300,1000)
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
        'x':0.17,
        'y':0.1,
        'z':0.64,
        'I_0' : 1000,
        'omega':1/10
        }

model.set_parameters(parameters)
result = model.compute(t)
#plt.plot(t,result['S'],label='S')
plt.plot(t,result['E'],label='E')
plt.plot(t,result['I_P'],label='I_P')
plt.plot(t,result['I_S'],label='I_S')
plt.plot(t,result['I_A'],label='I_A')
plt.legend()
plt.show()
