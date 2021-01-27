from deterministic_model import SIRTX
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
class range_analysis():
    """
    This class provides functions for analyzing varying parameter combinations.
    Parameter
    ---------
    Model, Parameter, Time
    """

    def __init__(self,model,parameter,time):
        self.model = model
        self.parameter = parameter
        self.time = time

    def range_result_plot(self,parameter_change, parameter_range, compartment):
        """
        Analysis and plot of varying values of a parameter.

        Parameter
        -----------
        Name of parameter (str)
        Varying values of this parameter (list)
        Compartment which is shown in a plot for the different values (str)

        Returns
        -----------
        Plot and results of all compartments for varying parameter (dict)
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

        color =  dict(mcolors.TABLEAU_COLORS, **mcolors.CSS4_COLORS)
        self.colors = [i for i in color.keys()]
        self.compartment = compartment
        self.parameter_change = parameter_change
        self.parameter_range = parameter_range

        results = {}
        for i in self.parameter_range:
            self.parameter.update({self.parameter_change:i})
            self.model.set_parameters(self.parameter)
            results[i]= self.model.compute(self.time)
            plt.plot(t,results[i][self.compartment],color = self.colors[self.parameter_range.index(i)],label = parameter_change + '='+ str(i))
        plt.legend()
        plt.xlabel('time [d]')
        plt.ylabel(compartment)
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
        color =  dict(mcolors.TABLEAU_COLORS, **mcolors.CSS4_COLORS)
        self.colors = [i for i in color.keys()]
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
                results[i][j] = self.model.compute(self.time)
                plt.plot(i,results[i][j][compartment].max(axis=0),'.', color = self.colors[list(self.parameter_range2).index(j)])

        lines = [Line2D([0], [0], color=self.colors[x], linewidth=3, linestyle='dotted') for x in range(len(self.parameter_range2))]
        labels = [(str(self.parameter_change2) + '=' + str(j)) for j in self.parameter_range2]
        plt.legend(lines,labels,loc='best')

        plt.xlabel(self.parameter_change1)
        plt.ylabel(compartment)
        plt.show()
        return results

class analysis_plot():
    """
    This class provides a plotting function for a given model with parameters and time a figure with
    individuals of all compartments in contrast to time.
    Parameter
    ---------
    Model, Parameter, Time

    Returns
    --------
    Plot
    """
    def __init__(self,model,parameter,time):

        self.model = model
        self.parameter = parameter
        self.time = time

    def result_plot(self):
        """
        Plotting function
        """

        color =  dict(mcolors.TABLEAU_COLORS, **mcolors.CSS4_COLORS)
        self.colors = [i for i in color.keys()]
        self.model.set_parameters(parameter)
        result = self.model.compute(self.time)
        for i in result.keys():
            plt.plot(t,result[i],color = self.colors[[i for i in result.keys()].index(i)],label = i)
        plt.legend()
        plt.xlabel('time [d]')
        plt.ylabel('individuals')
        plt.show()


if __name__=="__main__":
    population_size = 10000
    time = np.linspace(0,300,1000)
    model = SIRTX(N = population_size, t0 = 0)
    parameter = {
            'R0': 2.5,
            'q': 0.2,
            'a': 0.3,
            'rho' : 1/8,
            'k0' : 6.3,
            'I_0' : 1,
            'chi' : 1/2
            }
    q_range = [0,0.01,0.1,0.2]
    a_range = np.linspace(0,1,25)

    results = range_analysis(model,parameter,time).two_range_result_plot('a', a_range,'q',q_range,'I')
