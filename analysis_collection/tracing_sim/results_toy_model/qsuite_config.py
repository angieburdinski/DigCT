import os
import numpy as np

#=========== SIMULATION DETAILS ========
projectname = "DigCT"
basename = "deleting_edges"

seed = -1
N_measurements = 100

measurements = range(N_measurements)

phases_definitions = {
            'free': {
                    'tmaxs': [1e100],
                    'Rscale': [1.0],
                },
            'periodic lockdown': {
                    'tmaxs': [40,40,40,1e100],
                    'Rscale': [1.0,0.4,1.0,0.5],
                },
        }

phases = ['free', 'periodic lockdown']

N = 200000
k0 = 20
alpha = 1/3
rho = 1/9
R0 = 2.5
I0_prob = 0.001

q = 0.3
a_s = [0.0, 0.3, 0.5]

delete_edges_instead_of_scaling_R = True


external_parameters = [
                        (None, measurements),
                        ('phase', phases),
                      ]
internal_parameters = [
                        ('a', a_s),
                      ]
standard_parameters = [
                        ('phases', phases_definitions),
                        ('R0', R0),
                        ('k0', k0),
                        ('I0_prob', I0_prob),
                        ('q', q),
                        ('alpha',alpha),
                        ('rho',rho),
                        ('N', N),
                        ('delete_edges_instead_of_scaling_R', delete_edges_instead_of_scaling_R),
                      ]

only_save_times = False

#============== QUEUE ==================
queue = "SGE"
memory = "8G"
priority = 0

#============ CLUSTER SETTINGS ============
username =
server = 
useratserver = username + u'@' + server

shell = "/bin/bash"
pythonpath = "/opt/python36/bin/python3.6"
name = basename + "_N_meas_" + str(N_measurements)
serverpath = "/home/"+username +"/"+ projectname + "/" + name
resultpath = serverpath + "/results"

#============== LOCAL SETTINGS ============
localpath = os.path.join(os.getcwd(),"results_"+name)
n_local_cpus = 3

#========================
git_repos = [
                #( "/home/"+username+"/brownian-motion", pythonpath + " setup.py install --user" )
            ]
