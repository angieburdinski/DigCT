import epipack
import numpy as np
from bfmplot import pl as plt

class ParamDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self

class Tracing():
    def __init__(self, N, t0):
        self.t0 = t0
        self.N = N
        model = epipack.EpiModel(['S','E','I_P','I_S','I_A','R','T','X','Q'],initial_population_size = N)
        self.model = model

    def compute(self, t):
        result = self.model.integrate(t, integrator='dopri5')
        self.result = result
        return result

    def set_parameters(self,parameters):

        p = ParamDict(parameters)
        kappa = (p.q*p.rho)/(1-p.q)

        self.model.set_processes([
                    ('S','I_P',p.R0*p.beta/2,'E','I_P'),
                    ('S','I_A',p.R0*p.rho/4,'E','I_A'),
                    ('S','I_S',p.R0*p.rho/4,'E','I_S'),
                    ('E',p.alpha,'I_P'),
                    ('I_P',(1-p.x)*p.beta,'I_S'),
                    ('I_P',p.x)*p.beta,'I_A'),
                    ('I_A',p.rho,'R'),
                    ('I_S',p.rho,'R'),
                    ('I_S',kappa,'T'),
                    ('T',p.chi,'X'),
                    ('E','T',p.chi*p.y*p.a**2*p.k0*p.z,'T','T'),
                    ('I_P','T',p.chi*p.y*p.a**2*p.k0*p.z,'T','T'),
                    ('I_S','T',p.chi*p.y*p.a**2*p.k0*p.z,'T','T'),
                    ('I_A','T',p.chi*p.y*p.a**2*p.k0*p.z,'T','T'),
                    ('S','T',p.chi*p.a**2*p.k0*p.z,'Q','T'),
                    ('E','T',p.chi*(1-p.y)*p.a**2*p.k0*p.z,'X','T'),
                    ('I_P','T',p.chi*(1-p.y)*p.a**2*p.k0*p.z,'X','T'),
                    ('I_S','T',p.chi*(1-p.y)*p.a**2*p.k0*p.z,'X','T'),
                    ('I_A','T',p.chi*(1-p.y)*p.a**2*p.k0*p.z,'X','T'),
                    ('Q',p.omega,'S')])

        self.model.set_initial_conditions({'I_P': p.I_0, 'S': self.N - p.I_0})


if __name__=="__main__":
    population_size = 80e6
    t = np.linspace(0,365,1000)
    model = Tracing(N = population_size, t0 = 0)
    parameter = {
            'R0': 2.5,
            'q': ...,
            'a': ...,
            'chi':1/2.5,
            'rho' : 1/6,
            'alpha' : 1/2,
            'beta' : 1/2,
            'k0' : 6.3,
            'x':0.4,
            'y':0.1,
            'z':0.64,
            'I_0' : 1000,
            'omega':1/10
            }
    model.set_parameters(parameter)
    result = model.compute(t)
    plt.plot(t,result['S'],label = 'S')
    plt.plot(t,result['E'],label = 'E')
    plt.plot(t,result['I_P'],label = 'I_P')
    plt.plot(t,result['I_S'],label = 'I_S')
    plt.plot(t,result['I_A'],label = 'I_A')
    plt.plot(t,result['R'],label = 'R')
    plt.plot(t,result['T'],label = 'T')
    plt.plot(t,result['X'],label = 'X')
    plt.plot(t,result['Q'],label = 'Q')
    plt.legend()
    plt.xlabel('time [d]')
    plt.ylabel('individuals')
    plt.show()
