import os
import numpy as np
import epipack
from math import exp
from numpy import random
import networkx as nx
#=========== SIMULATION DETAILS ========
projectname = "paper_tracing"
basename = 'sw_clustered'
seed = -1
N_measurements = 100
measurements = range(N_measurements)
N = 200_000
q = 0.3
a = np.linspace(0,1,10)
y = 0.5
quarantiningS = True
clustered = [True,False]
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
                        ('clustered',clustered),

                      ]
internal_parameters = [
                        ('a', a),
                      ]
standard_parameters = [
                        ('q', q),
                        ('y', y),
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

#============== QUEUE ==================
username = "aburd"
server = "groot0.biologie.hu-berlin.de"
useratserver = username + u'@' + server

shell = "/bin/bash"
pythonpath = "/opt/python36/bin/python3.6"
name = basename
serverpath = "/home/"+username +"/"+ projectname + "/" + name
resultpath = serverpath + "/results"

#============== LOCAL SETTINGS ============
localpath = os.path.join(os.getcwd(),"results_"+name)
n_local_cpus = 7

#========================
git_repos = [

            ]
