import os
import numpy as np
#=========== SIMULATION DETAILS ========
projectname = "paper_tracing"
basename = "R0_analysis"

seed = -1
N_measurements = 100
measurements = range(N_measurements)
networks = ['ER','SW','EXP','WS-EXP']
R0 = np.linspace(0,10,51)
a = [0,0.3]
parameter = {
        'chi':1/2.5,
        'recovery_rate' : 1/7,
        'alpha' : 1/3,
        'beta' : 1/2,
        'number_of_contacts' : 20,
        'x':0.17,
        'I_0' : 200_000*0.01,
        'omega':1/10,
        "y" : 0.1,
        "z": 0.64,
        "N": 200_000,
        "quarantiningS": True,
        "q": 0.3,
        }
sampling_dt = 1
time = 10e6

external_parameters = [
                        ( None , measurements ),
                        ('a',a)
                      ]
internal_parameters = [
                        ('R0', R0),
                        ('networks', networks),

                      ]
standard_parameters = [ ('parameter', parameter ),
                        ('sampling_dt',sampling_dt),
                        ('time',time)
                      ]

only_save_times = False

#============== QUEUE ==================
queue = "SGE"
memory = "8G"
priority = -10

#============ CLUSTER SETTINGS ============
username =
server =
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
