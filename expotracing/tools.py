from deterministic_model import SIRX
import numpy as np
from bfmplot import pl as plt
from matplotlib import colors as mcolors

class range_analysis():

    def __init__(self,model,parameter):

        self.model = model
        self.parameter = parameter


    def range_result_plot(self,parameter_change,parameter_range, compartment):
        self.compartment = compartment
        self.parameter_change = parameter_change
        self.parameter_range = parameter_range
        """
        plots different results of I for different app-participation
        """

        results = {}
        for i in self.parameter_range:
            self.parameter.update({self.parameter_change:i})
            self.model.set_parameters(parameter)
            results[i]= self.model.compute(t)

        for i in self.parameter_range:
            plt.plot(t,results[i][compartment],label = parameter_change + '='+ str(i))
            plt.legend()
        plt.xlabel('time [d]')
        plt.ylabel(compartment)
        plt.show()
        return results

    def compare_tworange_result_plot(self,parameter_change1,parameter_range1,parameter_change2,parameter_range2, compartment):

        colorse = dict(mcolors.TABLEAU_COLORS, **mcolors.CSS4_COLORS)
        colors = [i for i in colorse.keys()]

        self.compartment = compartment
        self.parameter_change1 = parameter_change1
        self.parameter_range1 = parameter_range1
        self.parameter_change2 = parameter_change2
        self.parameter_range2 = parameter_range2
        """
        plots different results of I for different app-participation
        """

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
                results[i][j] = self.model.compute(t)
                #print(i, j,results[i][j][compartment].max(axis=0))
                plt.plot(i,results[i][j][compartment].max(axis=0),'.', color = colors[list(self.parameter_range2).index(j)])

        plt.legend()
        plt.xlabel(self.parameter_change1)
        plt.ylabel(compartment)
        plt.show()
        return results

if __name__=="__main__":
    population_size = 10000
    t = np.linspace(0,300,1000)
    model = SIRX(N = population_size, t0 = 0)
    parameter = {
            'R0': 2.5,
            'q': 0.2,
            'a': 0.3,
            'rho' : 1/8,
            'k0' : 50,
            'I_0' : 1,
            'chi' : 1/2
            }
    q_range = np.linspace(0,0.5,5)
    a_range = np.linspace(0,1,100)

    results = range_analysis(model,parameter).compare_tworange_result_plot('a', a_range,'q',q_range,'I')
