"""
Deterministic Models for Digital Exposure Tracing
"""
import epipack
import numpy as np
from bfmplot import pl as plt

class ParamDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self

class BaseModel():
    """Defines the BaseModel"""
    def __init__(self,model,t0):
        self.model = model
        self.t0 = t0

    def compute(self, t):
        """
        This function integrates the model over time t.
        t has to be list or numpy.ndarray of float.
        Returns
        -------
        result : dict
            The model result in the following format
            .. code:: python
            {
                    'S' : [ 10000, 99999, 99999.5, ...],
                    'I' : [ 1, 2, 2.5, ...],
                    ...
                }
        """
        result = self.model.integrate(t, integrator='dopri5')
        self.result = result
        return result

class SIRX(BaseModel):
    """
    This class describes a first generation tracing system without
    quarantining of susceptible contacts. It is based on the basic
    epidemic SIR system.
    """

    def __init__(self, N, t0):

        self.N = N
        model = epipack.EpiModel(['S','I','R','X'],initial_population_size = N)
        BaseModel.__init__(self, model, t0)

    def set_parameters(self,parameters):
        """
        Set the parameter values for the model
        Parameters
        ----------
        parameters : dict
                {
                    'R0' : ...,
                }
        """
        p = ParamDict(parameters)
        kappa = (p.q*p.rho)/(1-p.q)
        self.model.set_processes([
                    ('I', p.rho, 'R' ),
                    ('I', kappa, 'X' ),
                    ('I', 'S', p.R0*p.rho, 'I', 'I' ),
                    ('I', 'I', kappa * p.a**2 * p.k0, 'I', 'X' )]),
        self.model.set_initial_conditions({'I': p.I_0, 'S': self.N - p.I_0})

class SIRXQ(BaseModel):
    """
    This class describes a first generation tracing system with
    quarantining of susceptible contacts. It is based on the basic
    epidemic SIR system.
    """

    def __init__(self, N, t0):

        self.N = N
        model = epipack.EpiModel(['S','I','R','X','Q'],initial_population_size = N)
        BaseModel.__init__(self, model, t0)

    def set_parameters(self,parameters):
        """
        Set the parameter values for this model
        Parameters
        ----------
        parameters : dict
                {
                    'R0' : ...,
                }
        """
        p = ParamDict(parameters)
        kappa = (p.q*p.rho)/(1-p.q)
        self.model.set_processes([
                    ('I', p.rho, 'R' ),
                    ('I', kappa, 'X' ),
                    ('Q', p.omega, 'S' ),
                    ('I', 'S', p.R0*p.rho, 'I', 'I' ),
                    ('I', 'S', kappa * p.a**2 * p.k0, 'I', 'Q' ),
                    ('I', 'I', kappa * p.a**2 * p.k0, 'I', 'X' )])
        self.model.set_initial_conditions({'I': p.I_0, 'S': self.N - p.I_0})

class SIRTX(BaseModel):
    """
    This class describes a next generation tracing system without
    quarantining of susceptible contacts. It is based on the basic
    epidemic SIR system.
    """

    def __init__(self, N, t0):

        self.N = N
        model = epipack.EpiModel(['S','I','R','T','X'],initial_population_size = N)
        BaseModel.__init__(self, model, t0)

    def set_parameters(self,parameters):
        """
        Set the parameter values for this model
        Parameters
        ----------
        parameters : dict
                {
                    'R0' : ...,
                }
        """
        p = ParamDict(parameters)
        kappa = (p.q*p.rho)/(1-p.q)
        self.model.set_processes([
                            ('I', p.rho, 'R'),
                            ('I', kappa, 'T' ),
                            ('T', p.chi, 'X' ),
                            ('I', 'S', p.R0*p.rho, 'I', 'I'),
                            ('T', 'I', p.chi * p.a**2 * p.k0, 'T', 'T' )])
        self.model.set_initial_conditions({'I': p.I_0, 'S': self.N - p.I_0})

class SIRTXQ(BaseModel):
    """
    This class describes a next generation tracing system with
    quarantining of susceptible contacts. It is based on the basic
    epidemic SIR system.
    """

    def __init__(self, N, t0):

        self.N = N
        model = epipack.EpiModel(['S','I','R','T','X','Q'],initial_population_size=N)
        BaseModel.__init__(self, model, t0)

    def set_parameters(self,parameters):
        """
        Set the parameter values for this model
        Parameters
        ----------
        parameters : dict
                {
                    'R0' : ...,
                }
        """
        p = ParamDict(parameters)
        kappa = (p.q*p.rho)/(1-p.q)
        self.model.set_processes([
                    ('I', 'S', p.R0/p.rho, 'I', 'I' ),
                    ('T', 'S', p.chi * p.a**2 * p.k0, 'T', 'Q' ),
                    ('T', 'I', p.chi * p.a**2 * p.k0, 'T', 'T' ),
                    ('I', p.rho, 'R'),
                    ('I', kappa, 'T'),
                    ('T', p.chi, 'X'),
                    ('Q', p.omega, 'S')])

        self.model.set_initial_conditions({'I': p.I_0, 'S': self.N - p.I_0})

class FGE(BaseModel):

    def __init__(self, N, t0):

        self.N = N
        model = epipack.EpiModel(['S','E','I_P','I_S','I_A','R','X'],initial_population_size=N)
        BaseModel.__init__(self, model, t0)

    def set_parameters(self,parameters):
        """
        Set the parameter values for this model
        Parameters
        ----------
        parameters : dict
                {
                    'R0' : ...,
                }
        """
        p = ParamDict(parameters)
        kappa = (p.q*p.rho)/(1-p.q)
        kappa1 = 1/(1/(kappa)+p.delay))
        self.model.set_processes([
                    ('I_P', 'S', p.R0*p.beta/2, 'I_P', 'E' ),
                    ('I_A', 'S', p.R0*p.rho/4, 'I_P', 'E' ),
                    ('I_S', 'S', p.R0*p.rho/4, 'I_P', 'E' ),
                    ('E', p.alpha, 'I_P'),
                    ('I_P', 0.6*p.beta, 'I_S'),
                    ('I_P', 0.4*p.beta, 'I_A'),
                    ('I_S', kappa, 'X'),
                    ('I_S', 'E', kappa1 * p.a**2 * p.k0*0.64*0.1, 'I_S', 'X' ),
                    ('I_S', 'I_P', kappa1 * p.a**2 * p.k0*0.64*0.1, 'I_S', 'X' ),
                    ('I_S', 'I_A', kappa1 * p.a**2 * p.k0*0.64*0.1, 'I_S', 'X' ),
                    ('I_S', 'I_S', kappa1 * p.a**2 * p.k0*0.64*0.1, 'I_S', 'X' ),
                    ('I_A', p.rho, 'R'),
                    ('I_S', p.rho, 'R')])


        self.model.set_initial_conditions({'I_S': p.I_0, 'S': self.N - p.I_0})


if __name__=="__main__":
    population_size = 1000
    t = np.linspace(0,100,1000)
    model = FGE(N = population_size, t0 = 0)
    parameter = {
            'R0': 2.5,
            'q': 0.3,
            'a': 0.5,
            'delay':2.63,
            'rho' : 1/6,
            'alpha' : 1/2,
            'beta' : 1/2,
            'k0' : 6.3,
            'I_0' : 1
            }
    model.set_parameters(parameter)
    result = model.compute(t)
    plt.plot(t,result['S'],label = 'S')
    plt.plot(t,result['E'],label = 'E')
    plt.plot(t,result['I_P'],label = 'I_P')
    plt.plot(t,result['I_S'],label = 'I_S')
    plt.plot(t,result['I_A'],label = 'I_A')
    plt.plot(t,result['R'],label = 'R')
    plt.plot(t,result['X'],label = 'X')
    plt.legend()
    plt.xlabel('time [d]')
    plt.ylabel('individuals')
    plt.show()
    #plt.savefig('Try_1.pdf')
