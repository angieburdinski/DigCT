import os
import numpy as np
import epipack
from math import exp
from numpy import random
import networkx as nx
#=========== SIMULATION DETAILS ========
projectname = "paper_tracing"
basename = 'panels4_lockdown_ER_SW_200k'
#basename = 'panels4_lockdown_ER_SW'
seed = -1
N_measurements = 100
measurements = range(N_measurements)
N = 200_000
#N = 5_000
#N = 2_000
q = [0.1,0.3,0.5]
a = np.linspace(0,1,25)
y = 0.5
quarantiningS = True
clustered = False
lockdown = [True,False]
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
networks = ['ER','SW']

external_parameters = [
                        ( None, measurements ),
                        ('networks', networks),
                      ]
internal_parameters = [ ('q', q),
                        ('a', a),
                        ('lockdown', lockdown),
                      ]
standard_parameters = [
                        ('y', y),
                        ('N', N ),
                        ('quarantiningS', quarantiningS ),
                        ('parameter', parameter ),
                        ('sampling_dt',sampling_dt),
                        ('time',time),
                        ('clustered',clustered)
                      ]

only_save_times = False
#============== QUEUE ==================
queue = "SGE"
memory = "8G"
priority = -10

#============== QUEUE ==================
username =
server = 
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
