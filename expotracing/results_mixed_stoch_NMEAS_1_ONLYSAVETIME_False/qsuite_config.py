import os

#=========== SIMULATION DETAILS ========
projectname = "tracing"
basename = "mixed_stoch"

seed = -1
N_measurements = 1

measurements = range(N_measurements)

N = 10_000
k0 = 19
time = 10_000
parameter = {
        'R0': 2.0,
        'q': 0.5,
        'app_participation': 0.33,
        'chi':1/2.35,
        'recovery_rate' : 1/8,
        'alpha' : 1/3,
        'beta' : 1/2,
        'number_of_contacts' : 6.3,
        'x':0.17,
        'y':0.1,
        'z':0.64,
        'I_0' : 10,
        'omega':1/10
        }


external_parameters = [
                        ( None   , measurements ),
                      ]
internal_parameters = []
standard_parameters = [
                        ( 'N', N ),
                        ( 'k0', k0 ),
                        ( 'time', time ),
                        ( 'parameter', parameter ),

                      ]

only_save_times = False

#============== QUEUE ==================
queue = "SGE"
memory = "1G"
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
