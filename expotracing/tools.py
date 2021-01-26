from deterministic_model import SIRX
import numpy as np
from bfmplot import pl as plt

class app_range_analysis():

    def __init__(self,model,parameter,app_range):
        self.app_range = app_range
        self.model = model
        self.parameter = parameter

    def app_analysis_result(self):
        """
        saves different results for different app-participation in a dict with format:
        all_result = {
                    app-parti : {
                                'S':...
                                'I':...}
        }
        """
        all_result = {}
        for i in self.app_range:
            self.parameter.update({'a':i})
            self.model.set_parameters(parameter)
            result = self.model.compute(t)
            all_result[i]=result

        return all_result

    def app_analysis_I(self):
        """
        plots different results of I for different app-participation
        """
        all_result = app_range_analysis(model,parameter,app_range).app_analysis_result()
        for i in app_range:
            plt.plot(t,all_result[i]['I'],label = i)
            plt.legend()
        plt.show()




population_size = 1000
t = np.linspace(0,300,1000)
model = SIRX(N = population_size, t0 = 0)
parameter = {
        'R0': 2.5,
        'q': 0.3,
        'a': 0.5,
        'rho' : 1/8,
        'k0' : 50,
        'I_0' : 1,
        'chi' : 1/2
        }
app_range = np.linspace(0,1,6)

app_range_analysis(model,parameter,app_range).app_analysis_I()
