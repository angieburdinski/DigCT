from extendedmodel import (first_generation_tracing,next_generation_tracing,mixed_tracing)
import numpy as np
import matplotlib.pyplot as plt
from math import exp
import networkx as  nx
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
            print(sum(expected_degree_sequence)/self.N)
            return expected_degree_sequence


        expected_degree_sequence = seq(k,P)

        G = nx.configuration_model(expected_degree_sequence)
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))

        return G

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
    population_size = 80e6
    t = np.linspace(0,365,1000)
    model = mixed_tracing(N = population_size,quarantine_S_contacts = False)
    parameter = {
            'R0': 2.5,
            'q': 0.5,
            'app_participation': 0.33,
            'chi':1/2.5,
            'recovery_rate' : 1/6,
            'alpha' : 1/2,
            'beta' : 1/2,
            'number_of_contacts' : 6.3*1/0.33,
            'x':0.4,
            'y':0.1,
            'z':0.64,
            'I_0' : 1000,
            'omega':1/10
            }

    q = [0,0.01,0.2,0.4,0.6,0.8]
    number_of_contacts = np.linspace(0,100,50)
    #a = np.linspace(0,0.8,25)

    results = analysis(model,parameter,t).two_range_result('number_of_contacts',number_of_contacts,'q',q)
