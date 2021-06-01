from extendedmodel import (first_generation_tracing,next_generation_tracing,mixed_tracing)
import numpy as np
from math import exp
import networkx as  nx
import netwulf as nw
import networkx as nx
import epipack
from epipack.vis import visualize
from numpy import random
from smallworld import get_smallworld_graph

class ParamDict(dict):
    """

    """
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self

class configuration_network():
    """
    A Class to build a configuration network with an exponential degree
    distribution.

    Parameter
    ---------
    Number of individuals (int) : N
    Mean contact number (int) : k0
    """
    def __init__(self,N,k0):
        """
        Initalizes the configuration network
        """
        self.k0 = k0
        self.N = N

    def build(self):
        """
        This function builds the Network without selfloops or multiedges.

        Returns
        ------
        G (networkx object)
        """
        def expodegree(x):
            return 1/self.k0*exp(-x/self.k0)
        P = []
        k = []
        for i in range(self.N):
            p_k = expodegree(i)
            P.append(p_k)
            k.append(i)
        P = np.array(P)
        P /= P.sum()
        def seq(k,P):
            expected_degree_sequence = np.linspace(0,1,2)
            while sum(expected_degree_sequence) % 2 != 0:
                expected_degree_sequence = np.random.choice(
                  k,
                  self.N,
                  p = P
                )

            return expected_degree_sequence


        expected_degree_sequence = seq(k,P)

        G = nx.configuration_model(expected_degree_sequence)
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        print('average clustering'+str(nx.average_clustering(G)))
        print('info'+str(nx.info(G)))
        return G

class vis_mixed_config():
    def __init__(self,N,k0,t,parameter):
        self.N = N
        self.k0 = k0
        k_over_2 = int(self.k0/2)
        beta = 10e-9
        #beta = 10e-3 #for k = 20, N = 1000
        #beta = 10e-5 #for k = 20, N = 10_000
        #beta = 10e-6 #for k = 20, N = 20_000
        #beta = 10e-7 #for k = 20, N = 200_000
        self.G = get_smallworld_graph(self.N,k_over_2,beta)
        #self.G = configuration_network(self.N,self.k0).build()
        self.t = t
        self.edge_weight_tuples = [ (e[0], e[1], 1.0) for e in self.G.edges() ]
        self.k_norm = 2*len(self.edge_weight_tuples) /self.N
        self.model = epipack.StochasticEpiModel(['S','E','I_P','I_S','I_A','R','T','X','Sa','Ea','I_Pa','I_Sa','I_Aa','Ra','Ta','Xa','Qa'],self.N, self.edge_weight_tuples,directed=False)
        p = ParamDict(parameter)
        kappa = (p.q*p.recovery_rate)/(1-p.q)
        Sa0 = int(random.binomial(N-p.I_0, p.app_participation, 1))
        S0 = int(N - p.I_0 - Sa0)
        self.model.set_conditional_link_transmission_processes({

                ("Ta", "->", "Xa") : [
                        ("Xa", "I_Pa", p.y, "Xa", "Ta" ),
                        ("Xa", "I_Sa", p.y, "Xa", "Ta" ),
                        ("Xa", "I_Aa", p.y, "Xa", "Ta" ),
                        ("Xa", "Ea", p.y, "Xa", "Ta" ),
                        ("Xa", "Sa", "->", "Xa", "Qa" ),
                        ("Xa", "I_Pa", (1-p.y), "Xa", "Xa" ),
                        ("Xa", "I_Sa", (1-p.y), "Xa", "Xa" ),
                        ("Xa", "I_Aa", (1-p.y), "Xa", "Xa" ),
                        ("Xa", "Ea", (1-p.y), "Xa", "Xa" )]

                        })

        self.model.set_link_transmission_processes([

                    ('I_Pa','S',p.R0/self.k_norm*p.beta/2,'I_Pa','E'),
                    ('I_Aa','S',p.R0/self.k_norm*p.recovery_rate/2,'I_Aa','E'),
                    ('I_Sa','S',p.R0/self.k_norm*p.recovery_rate/2,'I_Sa','E'),

                    ('I_P','Sa',p.R0/self.k_norm*p.beta/2,'I_P','Ea'),
                    ('I_A','Sa',p.R0/self.k_norm*p.recovery_rate/2,'I_A','Ea'),
                    ('I_S','Sa',p.R0/self.k_norm*p.recovery_rate/2,'I_S','Ea'),

                    ('I_Pa','Sa',p.R0/self.k_norm*p.beta/2,'I_Pa','Ea'),
                    ('I_Aa','Sa',p.R0/self.k_norm*p.recovery_rate/2,'I_Aa','Ea'),
                    ('I_Sa','Sa',p.R0/self.k_norm*p.recovery_rate/2,'I_Sa','Ea'),

                    ('I_P','S',p.R0/self.k_norm*p.beta/2,'I_P','E'),
                    ('I_A','S',p.R0/self.k_norm*p.recovery_rate/2,'I_A','E'),
                    ('I_S','S',p.R0/self.k_norm*p.recovery_rate/2,'I_S','E')])

        self.model.set_node_transition_processes([
                    ('E',p.alpha,'I_P'),
                    ('I_P',(1-p.x)*p.beta,'I_S'),
                    ('I_P',p.x*p.beta,'I_A'),
                    ('I_A',p.recovery_rate,'R'),
                    ('I_S',p.recovery_rate,'R'),
                    ('I_S',kappa,'T'),
                    ('T',p.chi,'X'),
                    ('Qa',p.omega,'Sa'),
                    ('Ea',p.alpha,'I_Pa'),
                    ('I_Pa',(1-p.x)*p.beta,'I_Sa'),
                    ('I_Pa',p.x*p.beta,'I_Aa'),
                    ('I_Aa',p.recovery_rate,'Ra'),
                    ('I_Sa',p.recovery_rate,'Ra'),
                    ('I_Sa',kappa,'Ta'),
                    ('Ta',p.z*p.chi,'Xa'),
                    ('Ta',(1-p.z)*p.chi,'X')])

        self.model.set_network(self.N,self.edge_weight_tuples)
        self.model.set_random_initial_conditions({ 'Sa' : Sa0,'S' : S0,'I_P':p.I_0})
        stylized_network, config = nw.visualize(self.G)
        visualize(self.model, stylized_network, sampling_dt=0.1,ignore_plot_compartments=['S','Sa'],quarantine_compartments=['X',"Xa","R","Ra"],)

class analysis():
    """
    This class provides functions to analyze a model.

    Parameter
    ---------
    Model, Parameter (dict)

    Example
    -------
    N = 80e6
    t = np.linspace(0,365,1000)
    model = first_generation_tracing(N, quarantine_S_contacts = False)
    parameter = {
            'R0': ...,
        }

    analysis(model,parameter)
    """

    def __init__(self,model,parameter):

        self.model = model
        self.parameter = parameter


    def range_result(self,parameter_change, parameter_range,t):
        """
        Fuction to analyse the chosen model for varying values of a parameter.

        Parameter
        -----------
        Name of varying parameter (str) : parameter_change
        Values of this parameter (list/array) : parameter_range
        Time (list/array) : t

        Returns
        -----------
        Results of all compartments for varying parameter (dict)
        Example:

        Results = {
        0.1 : {
                'S':...,
                'I':...,
                }
        ,
        0.2 : {
                'S':...,
                'I':...,
                }
        }
        """
        self.parameter_change = parameter_change
        self.parameter_range = parameter_range
        self.t = t
        results = {}
        for i in self.parameter_range:
            self.parameter.update({self.parameter_change:i})
            self.model.set_parameters(self.parameter)
            results[i]= self.model.compute(self.t)
        return results

    def stoch_range_result(self,parameter_change, parameter_range,time):
        """
        Fuction to analyse the chosen stochastic model for varying values of a parameter.

        Parameter
        -----------
        Name of varying parameter (str) : parameter_change
        Values of this parameter (list/array) : parameter_range
        Time (int) : time

        Returns
        -----------
        Time (list), results of all compartments for varying parameter (dict)
        Example:

        t (list), result = {
        0.1 : {
                'S':...,
                'I':...,
                }
        ,
        0.2 : {
                'S':...,
                'I':...,
                }
        }
        """
        self.time = time
        self.parameter_change = parameter_change
        self.parameter_range = parameter_range
        results = {}
        for i in self.parameter_range:
            self.parameter.update({self.parameter_change:i})
            self.model.set_parameters(self.parameter)
            t,results[i]= self.model.compute(self.time)
        return t,results

    def two_range_result(self,parameter_change1,parameter_range1,parameter_change2,parameter_range2,t):
        """
        Analysis of varying values of two parameters.

        Parameter
        -----------
        Name of first parameter (str) : parameter_change1
        Varying values of this first parameter (list/array) : parameter_range1
        Name of second parameter (str) : parameter_change2
        Varying values of this second parameter (list/array) : parameter_range2
        Time (list/array) : t

        Returns
        -----------
        Results of all compartments for varying parameter (dict)
        Example:
        Results = {
        0.1 : {
                0.0 : {
                        'S':...,
                        'I':...,},
                0.1 : {
                        'S':...,
                        'I':...,}
                        },
        0.2 : {
                0.0 : {
                        'S':...,
                        'I':...,},
                0.1 : {
                        'S':...,
                        'I':...,}
                        }
        }
        """

        self.t = t
        self.parameter_change1 = parameter_change1
        self.parameter_range1 = parameter_range1
        self.parameter_change2 = parameter_change2
        self.parameter_range2 = parameter_range2


        results = {}
        for i in self.parameter_range1:
            results[i] = {}
            for j in self.parameter_range2:
                results[i][j] = {}

        for i in self.parameter_range1:
            for j in self.parameter_range2:
                self.parameter.update({self.parameter_change1:i})
                self.parameter.update({self.parameter_change2:j})
                self.model.set_parameters(self.parameter)
                results[i][j] = self.model.compute(self.t)
        return results

    def stoch_two_range_result(self,parameter_change1,parameter_range1,parameter_change2,parameter_range2,time):
            """
            Analysis of varying values of two parameters.

            Parameter
            -----------
            Name of first parameter (str) : parameter_change1
            Varying values of this first parameter (list/array) : parameter_range1
            Name of second parameter (str) : parameter_change2
            Varying values of this second parameter (list/array) : parameter_range2
            Time (int) : time

            Returns
            -----------
            Time (list) and results of all compartments for varying parameter (dict)
            Example:
            t (list) , result = {
            0.1 : {
                    0.0 : {
                            'S':...,
                            'I':...,},
                    0.1 : {
                            'S':...,
                            'I':...,}
                            },
            0.2 : {
                    0.0 : {
                            'S':...,
                            'I':...,},
                    0.1 : {
                            'S':...,
                            'I':...,}
                            }
            }
            """


            self.parameter_change1 = parameter_change1
            self.parameter_range1 = parameter_range1
            self.parameter_change2 = parameter_change2
            self.parameter_range2 = parameter_range2
            self.time = time

            results = {}
            for i in self.parameter_range1:
                results[i] = {}
                for j in self.parameter_range2:
                    results[i][j] = {}

            for i in self.parameter_range1:
                for j in self.parameter_range2:
                    self.parameter.update({self.parameter_change1:i})
                    self.parameter.update({self.parameter_change2:j})
                    self.model.set_parameters(self.parameter)
                    t, results[i][j] = self.model.compute(self.time)
            return t, results





if __name__=="__main__":
    population_size = 10_000
    t = np.linspace(0,365,1000)
    model = mixed_tracing(N = population_size,quarantine_S_contacts = False)
    parameter = {
            'R0': 2.5,
            'q': 0.5,
            'app_participation': 0.3,
            #'chi':1,
            'chi':1/2.5,
            'recovery_rate' : 1/7,
            'alpha' : 1/3,
            'beta' : 1/2,
            'number_of_contacts' : 6.3,
            'x':0.17,
            'y':0.1,
            'z':0.64,
            'I_0' : 10,
            'omega':1/10
            }

    q = [0,0.01,0.2,0.4,0.6,0.8]
    a = np.linspace(0,0.8,25)
    vis_mixed_config(N = 1000, k0 = 20, t = t, parameter = parameter)
    #results = analysis(model,parameter).two_range_result('app_participation',a,'q',q,t)
