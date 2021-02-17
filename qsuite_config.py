import os
import numpy as np
#=========== SIMULATION DETAILS ========
projectname = "expotracing"
basename = "config"

seed = -1
N_measurements = 100

measurements = range(N_measurements)
N1 = 4039
eta1  = 1/8 * 2.5
q1 = [0,0.01,0.1,0.2,0.35,0.5]
rho1 = 1/8
a1 = np.linspace(0,1,50)
chi1 = 1/2

alpha1 = 1/2



external_parameters = [


                        ( None   , measurements ),
                      ]
internal_parameters = [
                        ('a', a1),
                        ( 'q', q1  ),

                      ]
standard_parameters = [
                        ( 'N', N1 ),
                        ( 'rho', rho1 ),
                        ( 'alpha', alpha1 ),
                        ( 'chi', chi1 ),
                        ( 'eta', eta1 ),

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
