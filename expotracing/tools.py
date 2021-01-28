from extendedmodel import (first_generation_tracing,next_generation_tracing,mixed_tracing)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D

class analysis():
    """
    This class provides functions to analyse a model and plot the chosen analysis.

    Parameter
    ---------
    Model, Parameter (dict), Time (list/array)

    Example
    -------
    N = 80e6
    t = np.linspace(0,365,1000)
    model = first_generation_tracing(N, quarantine_S_contacts = False)
    parameter = {
            'R0': ...,
        }

    analysis(model,parameter,t)
    """

    def __init__(self,model,parameter,t):

        self.model = model
        self.parameter = parameter
        self.t = t
        color =  dict(mcolors.TABLEAU_COLORS, **mcolors.CSS4_COLORS)
        self.colors = [i for i in color.keys()]

    def basic(self):
        """
        Function that plots number of individuals in each
        compartment in contrast to increasing time.
        """
        self.model.set_parameters(self.parameter)
        result = self.model.compute(self.t)
        for i in result.keys():
            plt.plot(t,result[i],color = self.colors[[i for i in result.keys()].index(i)],label = i)
        plt.legend()
        plt.xlabel('time [d]')
        plt.ylabel('individuals')
        plt.show()

    def range_result(self,parameter_change, parameter_range, compartments):
        """
        Fuction to analyse the chosen model for varying values of a parameter.

        Parameter
        -----------
        Name of varying parameter (str)
        values of this parameter (list/array)

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
        self.compartments = compartments
        fig, axs = plt.subplots(len(compartments),sharex=True, sharey=True )
        results = {}
        for i in self.parameter_range:
            self.parameter.update({self.parameter_change:i})
            self.model.set_parameters(self.parameter)
            results[i]= self.model.compute(self.t)

            for x in range(len(self.compartments)):
                axs[x].plot(self.t,results[i][self.compartments[x]], color = self.colors[list(self.parameter_range).index(i)],label = self.parameter_change + '='+ str(i))
                axs[x].set_xlabel('time [d]')
                axs[x].set_ylabel(self.compartments[x])
        plt.legend()
        plt.show()

        return results

    def two_range_result_plot(self,parameter_change1,parameter_range1,parameter_change2,parameter_range2, compartment):
        """
        Analysis and plot of varying values of two parameters.

        Parameter
        -----------
        Name of first parameter (str)
        Varying values of this first parameter (list/array)
        Name of second parameter (str)
        Varying values of this second parameter (list/array)
        Compartment which is shown in a plot for the different values (str)

        Returns
        -----------
        Plot and results of all compartments for varying parameter (dict)
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

        self.compartment = compartment
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
                self.model.set_parameters(parameter)
                results[i][j] = self.model.compute(self.t)
                plt.plot(i,results[i][j][compartment].max(axis=0),'.', color = self.colors[list(self.parameter_range2).index(j)])

        lines = [Line2D([0], [0], color=self.colors[x], linewidth=3, linestyle='dotted') for x in range(len(self.parameter_range2))]
        labels = [(str(self.parameter_change2) + '=' + str(j)) for j in self.parameter_range2]
        plt.legend(lines,labels,loc='best')
        plt.xlabel(self.parameter_change1)
        plt.ylabel(compartment)
        plt.show()
        return results

    def I_red(self):
        pass

    def RX(self):
        pass

    def Q(self):
        pass



if __name__=="__main__":
    population_size = 80e6
    t = np.linspace(0,365,1000)
    model = first_generation_tracing(N = population_size,quarantine_S_contacts = False)
    parameter = {
            'R0': 2.5,
            'q': 0.5,
            'app_participation': 0.3,
            'chi':1/2.5,
            'recovery_rate' : 1/6,
            'alpha' : 1/2,
            'beta' : 1/2,
            'number_of_contacts' : 6.3,
            'x':0.6,
            'y':0.1,
            'z':0.64,
            'I_0' : 1000,
            'omega':1/10
            }

    q = np.linspace(0,1,10)
    a = np.linspace(0,1,50)

    results = analysis(model,parameter,t).range_result('a',a,['I','S'])
