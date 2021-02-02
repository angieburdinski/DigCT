from extendedmodel import (first_generation_tracing,next_generation_tracing,mixed_tracing,stoch_mixed_tracing)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
import networkx as nx

class plot():

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
            plt.plot(self.t,result[i],color = self.colors[[i for i in result.keys()].index(i)],label = i)
        plt.legend()
        plt.xlabel('time [d]')
        plt.ylabel('individuals')
        plt.show()

    def range_plot(self,parameter_change, parameter_range, compartments):
        """
        Fuction to plot the chosen model for varying values of a parameter.

        Parameter
        -----------
        Name of varying parameter (str)
        Values of this parameter (list/array)
        Compartments  (list)

        """
        self.parameter_change = parameter_change
        self.parameter_range = parameter_range
        self.compartments = compartments
        fig, axs = plt.subplots(1,len(compartments),figsize = (len(compartments)*3,3),sharex=True, sharey=True )
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

    def two_range_plot(self,parameter_change1,parameter_range1,parameter_change2,parameter_range2, compartments):
        """
        Plot of varying values of two parameters and plot the results of the compartments .

        Parameter
        -----------
        Name of first parameter (str)
        Varying values of this first parameter (list/array)
        Name of second parameter (str)
        Varying values of this second parameter (list/array)
        Compartments which are shown in a plot for the different values (list)

        """

        self.compartments = compartments
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
                t, results[i][j] = self.model.compute(self.t)

        fig, axs = plt.subplots(1,len(self.compartments),figsize = (len(compartments)*3,3), sharex=True, sharey=True )

        for y in self.compartments:
            for i in self.parameter_range1:
                for j in self.parameter_range2:
                    axs[list(self.compartments).index(y)].plot(i,results[i][j][y].max(axis=0),'.', color = self.colors[list(self.parameter_range2).index(j)])
            axs[list(self.compartments).index(y)].set_xlabel(self.parameter_change1)
            axs[list(self.compartments).index(y)].set_ylabel(y)
        lines = [Line2D([0], [0], color=self.colors[x], linewidth=3, linestyle='dotted') for x in range(len(self.parameter_range2))]
        labels = [(str(self.parameter_change2) + '=' + str(j)) for j in self.parameter_range2]
        fig.legend(lines, labels)
        plt.show()
if __name__=="__main__":
    N = 10
    k0 = 2
    G = nx.barabasi_albert_graph(N,k0)
    t=1000

    model = stoch_mixed_tracing(G,quarantine_S_contacts = False)
    parameter = {
            'R0': 2.5,
            'q': 0.5,
            'app_participation': 0.33,
            'chi':1/2.5,
            'recovery_rate' : 1/6,
            'alpha' : 1/2,
            'beta' : 1/2,
            #'number_of_contacts' : 6.3*1/0.33,
            'x':0.83,
            'y':0.1,
            'z':0.64,
            'I_0' : 1,
            'omega':1/10
            }

    q = [0,0.01,0.2,0.4,0.6,0.8]
    #number_of_contacts = np.linspace(0,100,50)
    a = np.linspace(0,0.8,5)

    results = plot(model,parameter,t).two_range_plot('app_participation',a,'q',q,['Ra','Xa','R','X'])
