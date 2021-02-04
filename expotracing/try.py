import netwulf as nw
import networkx as nx
import epipack
#from tools import configuration_network
from tools import vis_mixed_config
from epipack.vis import visualize
from numpy import random

N = 500
k0 = 19
t = 1000
parameter = {
        'R0': 2.5,
        'q': 0.5,
        'app_participation': 0.33,
        'chi':1/2.5,
        'recovery_rate' : 1/6,
        'alpha' : 1/2.5,
        'beta' : 1/2.5,
        #'number_of_contacts' : 6.3,
        'x':0.17,
        'y':0.1,
        'z':0.64,
        'I_0' : 10,
        'omega':1/10
        }
vis_mixed_config(N,k0,t,parameter)
