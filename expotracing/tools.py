from deterministic_model import FGE
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
class change(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self

class analysis():

    """
    This class provides functions to analyse a model.

    Parameter
    ---------
    Model, Parameter, Time

    """

    def __init__(self,model,parameter,time):
        self.model = model
        self.parameter = parameter
        self.time = time
        color =  dict(mcolors.TABLEAU_COLORS, **mcolors.CSS4_COLORS)
        self.colors = [i for i in color.keys()]

    def result_plot(self):
        """
        Function that plots number of individuals in each
        compartment in contrast to increasing time.
        """

        self.model.set_parameters(parameter)
        result = self.model.compute(self.time)
        for i in result.keys():
            plt.plot(self.time,result[i],color = self.colors[[i for i in result.keys()].index(i)],label = i)
        plt.legend()
        plt.xlabel('time [d]')
        plt.ylabel('individuals')
        plt.show()

    def range_result(self,parameter_change, parameter_range,compartments):
        """
        Fuction to analyse the chosen model for varying values of a parameter.
        Plots the results for chosen compartments

        Parameter
        -----------
        Name of varying parameter (str)
        values of this parameter (list/array)
        Compartments (list)


        Returns
        -----------
        Results of all compartments for varying parameter (dict)
        and plot of the results for the chosen compartment for varying parameter.

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
            results[i]= self.model.compute(self.time)

            for x in range(len(self.compartments)):
                axs[x].plot(self.time,results[i][self.compartments[x]], color = self.colors[list(self.parameter_range).index(i)],label = self.parameter_change + '='+ str(i))

            #plt.plot(self.time,results[i][self.compartment],color = self.colors[list(self.parameter_range).index(i)],label = parameter_change + '='+ str(i))

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
                results[i][j] = self.model.compute(self.time)
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
    population_size = 10000
    time = np.linspace(0,200,1000)
    model = FGE(N = population_size, t0 = 0)
    parameter = {
            'R0': 1.5,
            'q': 0.3,
            'a': 0.25,
            'rho' : 1/6,
            'alpha' : 1/2,
            'beta' : 1/2,
            'delay':2.63,
            'k0' : 6.3,
            'I_0' : 1
            }
    #a = np.linspace(0,1,100)
    q = np.linspace(0,1,5)
    a = np.linspace(0,1,30)
    results = analysis(model,parameter,time).two_range_result_plot('a',a,'q',q, 'I_S')
