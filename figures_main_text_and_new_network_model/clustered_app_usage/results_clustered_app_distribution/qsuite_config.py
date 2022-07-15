import os
import numpy as np
import epipack
from math import exp
from numpy import random
import networkx as nx
#=========== SIMULATION DETAILS ========
projectname = "paper_tracing"
basename = 'clustered_app_distribution'

seed = -1
N_measurements = 100
measurements = range(N_measurements)
networks = ['ER','SW','ER_exp','SW_exp']
a = np.linspace(0,1,10)
clustered = [True,False]
sampling_dt = 1
time = 10e6
N = 200_000

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
        "q": 0.3,
        "y": 0.5,
        }

external_parameters = [
                        ( None, measurements ),
                      ]
internal_parameters = [ ('networks', networks),
                        ('a', a),
                        ('clustered',clustered)]
standard_parameters = [ ('N', N ),
                        ('parameter', parameter ),
                        ('sampling_dt',sampling_dt),
                        ('time',time),
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
