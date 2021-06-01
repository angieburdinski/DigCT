import os
import numpy as np

#=========== SIMULATION DETAILS ========
projectname = "DigCT"
basename = "deleting_edges_50"

seed = -1
N_measurements = 100

measurements = range(N_measurements)

phases_definitions = {
            'free': {
                    'tmaxs': [1e100],
                    'Rscale': [1.0],
                },
            'periodic lockdown': {
                    'tmaxs': [50,50,50,1e100],
                    'Rscale': [1.0,0.4,1.0,0.5],
                },
        }

phases = ['free', 'periodic lockdown']

N = 200000
rho =  1/7
chi = 1/2.5
alpha = 1/3
beta = 1/2
k0 =  20
x = 0.17
I0_prob = 0.001
omega = 1/10
z = 0.64
R0 = 2.5
y = 0.1
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
                        ('chi', chi),
                        ('beta',beta),
                        ('k0',k0),
                        ('x',x),
                        ('y',y),
                        ('z',z),
                        ('omega',omega),
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
username = "aburd"
server = "groot0.biologie.hu-berlin.de"
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
