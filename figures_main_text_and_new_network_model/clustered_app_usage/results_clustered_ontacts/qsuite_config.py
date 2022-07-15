import os
import numpy as np

#=========== SIMULATION DETAILS ========
projectname = "paper_tracing"
basename = 'clustered_ontacts'
seed = -1
N_measurements = 100
measurements = range(N_measurements)
N = 200_000
a = np.linspace(0,1,10)
networks = ['ER','SW','ER_exp','SW_exp']
number_of_contacts = 20
clustered = [True, False]
external_parameters = [
                        ( None , measurements ),

                    ]
internal_parameters = [
                        ( 'a' , a ),
                        ( 'clustered' , clustered),
                        ( 'network' , networks),
                       ]
standard_parameters = [
                        ( 'N' , N ),
                        ( 'number_of_contacts' , number_of_contacts)
                      ]
only_save_times = False
#============== QUEUE ==================
queue = "SGE"
memory = "8G"
priority = 0
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
git_repos = []
