import os
import numpy as np
import epipack
from math import exp
from numpy import random
import networkx as nx
#=========== SIMULATION DETAILS ========
projectname = "paper_tracing"
basename = "smallworld_withoutQ_v2"

seed = -1
N_measurements = 100
measurements = range(N_measurements)
N = 200_000
q = [0,0.1,0.3,0.5]
a = np.linspace(0,1,25)
y = [0.1]
quarantiningS = False
parameter = {
        'chi':1/2.5,
        'recovery_rate' : 1/7,
        'alpha' : 1/3,
        'beta' : 1/2,
        'number_of_contacts' : 20,
        'x':0.17,
        'I_0' : N*0.01,
        'omega':1/10,
        "z": 0.64,
        "R0": 2.5,
        }

sampling_dt = 1
time = 10e6


external_parameters = [
                        ( None , measurements ),
                      ]
internal_parameters = [
                        ('a', a),
                        ('q', q),
                        ('y', y),
                      ]
standard_parameters = [
                        ('N', N ),
                        ('quarantiningS', quarantiningS ),
                        ('parameter', parameter ),
                        ('sampling_dt',sampling_dt),
                        ('time',time)
                      ]

only_save_times = False

#============== QUEUE ==================
queue = "SGE"
memory = "8G"
priority = -10

#============ CLUSTER SETTINGS ============
username = "aburd"
server = "groot0.biologie.hu-berlin.de"
useratserver = username + u'@' + server

shell = "/bin/bash"
pythonpath = "/opt/python36/bin/python3.6"
name = basename + "_NMEAS_" + str(N_measurements) + "_ONLYSAVETIME_" + str(only_save_times)
serverpath = "/home/"+username +"/"+ projectname + "/" + name
resultpath = serverpath + "/results"

#============== LOCAL SETTINGS ============
localpath = os.path.join(os.getcwd(),"results_"+name)
n_local_cpus = 1

#========================
git_repos = []
