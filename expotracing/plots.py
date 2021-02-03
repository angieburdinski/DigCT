from extendedmodel import (first_generation_tracing,next_generation_tracing,mixed_tracing,stoch_mixed_tracing)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
import networkx as nx

class plot():

    def __init__(self,model,parameter):

        self.model = model
        self.parameter = parameter
        color =  dict(mcolors.TABLEAU_COLORS, **mcolors.CSS4_COLORS)
        self.colors = [i for i in color.keys()]

    def basic(self,t):
        """
        Function that plots number of individuals in each
        compartment in contrast to increasing time.

        Parameter
        ----------
        Time (list/array) : t
        """
        self.t = t
        self.model.set_parameters(self.parameter)
        result = self.model.compute(self.t)
        for i in result.keys():
            plt.plot(self.t,result[i],color = self.colors[[i for i in result.keys()].index(i)],label = i)
        plt.legend()
        plt.xlabel('time [d]')
        plt.ylabel('individuals')
        plt.show()

    def stoch_basic(self,time):
        """
        Function that plots number of individuals in each
        compartment in contrast to increasing time.

        Parameter
        ----------
        Time (int) : time
        """
        self.time = time
        self.model.set_parameters(self.parameter)
        t, result = self.model.compute(self.time)
        for i in result.keys():
            plt.plot(t,result[i],color = self.colors[[i for i in result.keys()].index(i)],label = i)
        plt.legend()
        plt.xlabel('time [d]')
        plt.ylabel('individuals')
        plt.show()

    def range_plot(self,parameter_change, parameter_range, compartments,t):
        """
        Fuction to plot the chosen model for varying values of a parameter.

        Parameter
        -----------
        Name of varying parameter (str) : parameter_change
        Values of this parameter (list/array) : parameter_range
        Compartments  (list) : compartments
        Time (list/array) : t

        """
        self.parameter_change = parameter_change
        self.parameter_range = parameter_range
        self.compartments = compartments
        self.t = t
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

    def stoch_range_plot(self,parameter_change, parameter_range, compartments,time):
            """
            Fuction to plot the chosen model for varying values of a parameter.

            Parameter
            -----------
            Name of varying parameter (str) : parameter_change
            Values of this parameter (list/array) : parameter_range
            Compartments  (list) : compartments
            Time (int) : time

            """
            self.parameter_change = parameter_change
            self.parameter_range = parameter_range
            self.compartments = compartments
            self.time = time
            fig, axs = plt.subplots(1,len(compartments),figsize = (len(compartments)*3,3),sharex=True, sharey=True )
            results = {}
            for i in self.parameter_range:
                self.parameter.update({self.parameter_change:i})
                self.model.set_parameters(self.parameter)
                t, results[i]= self.model.compute(self.time)
                for x in range(len(self.compartments)):
                    axs[x].plot(t,results[i][self.compartments[x]], color = self.colors[list(self.parameter_range).index(i)],label = self.parameter_change + '='+ str(i))
                    axs[x].set_xlabel('time [d]')
                    axs[x].set_ylabel(self.compartments[x])
            plt.legend()
            plt.show()

    def two_range_plot(self,parameter_change1,parameter_range1,parameter_change2,parameter_range2, compartments,t):
        """
        Plot of varying values of two parameters and plot the results of the compartments .

        Parameter
        -----------
        Name of first parameter (str) : parameter_change1
        Varying values of this first parameter (list/array) : parameter_range1
        Name of second parameter (str) : parameter_change2
        Varying values of this second parameter (list/array) : parameter_range2
        Compartments which are shown in a plot for the different values (list) : compartments
        Time (list/array) : t

        """
        self.t = t
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
                results[i][j] = self.model.compute(self.t)

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

    def stoch_two_range_plot(self,parameter_change1,parameter_range1,parameter_change2,parameter_range2, compartments,time):
            """
            Plot of varying values of two parameters and plot the results of the compartments .

            Parameter
            -----------
            Name of first parameter (str) : parameter_change1
            Varying values of this first parameter (list/array) : parameter_range1
            Name of second parameter (str) : parameter_change2
            Varying values of this second parameter (list/array) : parameter_range2
            Compartments which are shown in a plot for the different values (list) : compartments
            Time (int) : time
            """
            self.time = time
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
                    t,results[i][j] = self.model.compute(self.time)

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
    N = 1000
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
            'I_0' : 10,
            'omega':1/10
            }

    q = [0,0.01,0.2,0.4,0.6,0.8]
    #number_of_contacts = np.linspace(0,100,50)
    a = np.linspace(0,0.8,20)

    results = plot(model,parameter,t).two_range_plot('app_participation',a,'q',q,['Ra','Xa','R','X'])
