"""
Deterministic Models for Digital Exposure Tracing
"""
import epipack
import numpy as np
import matplotlib.pyplot as plt
class ParamDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self

class BaseModel():
    """Defines the BaseModel"""
    integrator = 'dopri5'
    def __init__(self,epimodel,t0,):
        self.epimodel = epimodel
        self.t0 = t0

    def set_parameters(self,parameters):
        """
        This function takes a parameter dictionary
        and updates the numerical ODEs accordingly.
        This method is supposed to be overridden by
        child classes.
        """
        pass

    def compute(self, time_points=None):
        """
        This function carries out the model integration
        given the parameters that have been set.
        Parameters
        ----------
        time_points : list or numpy.ndarray of float, default: None
            An ordered iterable that contains
            the time points at which the model
            should be evaluated
        Returns
        -------
        result : dict
            The model result in the following format
            .. code:: python
            {
                'compartments' : {
                    'S' : [ 10000, 99999, 99999.5, ...],
                    'I' : [ 1, 2, 2.5, ...],
                    ...
                }
            }

        """

        if time_points is None:
            if self.time_points is not None:
                time_points = self.time_points
            else:
                raise ValueError('integration time points are not set')

        result = self.epimodel.integrate(time_points, integrator=self.integrator)
        result = {
                    'compartments': result
                 }

        self.result = result

        return result


class SIRX(BaseModel):
    """
    This class describes a first generation tracing system without
    quarantining of susceptible contacts. It is based on the basic
    epidemic SIR system.
    """

    def __init__(self,population_size, t0):

        self.population_size = population_size

        model = epipack.EpiModel(['S','I','R','X'],initial_population_size=population_size)

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
        self.epimodel.set_processes([
                    ('I', p.rho, 'R' ),
                    ('I', kappa, 'X' ),
                    ('I', 'S', p.R0*p.rho, 'I', 'I' ),
                    ('I', 'I', kappa * p.a**2 * p.k0, 'I', 'X' )]),

        self.epimodel.set_initial_conditions({'I': p.I_0, 'S': self.population_size - p.I_0})
class SIRXQ(BaseModel):
    """
    This class describes a first generation tracing system without
    quarantining of susceptible contacts. It is based on the basic
    epidemic SIR system.
    """

    def __init__(self,population_size, t0):

        self.population_size = population_size

        model = epipack.EpiModel(['S','I','R','X','Q'],initial_population_size=population_size)

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
        self.epimodel.set_processes([
                    ('I', p.rho, 'R' ),
                    ('I', kappa, 'X' ),
                    ('Q', p.omega, 'S' ),
                    ('I', 'S', p.R0*p.rho, 'I', 'I' ),
                    ('I', 'S', kappa * p.a**2 * p.k0, 'I', 'Q' ),
                    ('I', 'I', kappa * p.a**2 * p.k0, 'I', 'X' )])

        self.epimodel.set_initial_conditions({'I': p.I_0, 'S': self.population_size - p.I_0})

class SIRTX(BaseModel):
    """
    This class describes a first generation tracing system without
    quarantining of susceptible contacts. It is based on the basic
    epidemic SIR system.
    """

    def __init__(self,population_size, t0):

        self.population_size = population_size

        model = epipack.EpiModel(['S','I','R','T','X'],initial_population_size=population_size)

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
        self.epimodel.set_processes([
                            ('I', p.rho, 'R'),
                            ('I', kappa, 'T' ),
                            ('T', p.chi, 'X' ),
                            ('I', 'S', p.R0*p.rho, 'I', 'I'),
                            ('T', 'I', p.chi * p.a**2 * p.k0, 'T', 'T' )])

        self.epimodel.set_initial_conditions({'I': p.I_0, 'S': self.population_size - p.I_0})

class SIRTXQ(BaseModel):
    """
    This class describes a first generation tracing system without
    quarantining of susceptible contacts. It is based on the basic
    epidemic SIR system.
    """

    def __init__(self,population_size, t0):

        self.population_size = population_size

        model = epipack.EpiModel(['S','I','R','T','X','Q'],initial_population_size=population_size)

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
        self.epimodel.set_processes([
                    ('I', 'S', p.R0/p.rho, 'I', 'I' ),
                    ('T', 'S', p.chi * p.a**2 * p.k0, 'T', 'Q' ),
                    ('T', 'I', p.chi * p.a**2 * p.k0, 'T', 'T' ),
                    ('I', p.rho, 'R'),
                    ('I', kappa, 'T'),
                    ('T', p.chi, 'X'),
                    ('Q', p.omega, 'S')])

        self.epimodel.set_initial_conditions({'I': p.I_0, 'S': self.population_size - p.I_0})

if __name__=="__main__":
    N = 1000
    t = np.linspace(0,300,1000)
    model = SIRX(population_size = N, t0 = 0)
    P = {
            'R0': 2.5,
            'q': 0.3,
            'a': 0.5,
            'rho' : 1/8,
            'k0' : 50,
            'I_0' : 1
            }
    model.set_parameters(P)
    result = model.compute(t)
    plt.plot(t,result['compartments']['S'],label = 'S')
    plt.plot(t,result['compartments']['I'],label = 'I')
    plt.plot(t,result['compartments']['R'],label = 'R')
    plt.plot(t,result['compartments']['X'],label = 'X')
    plt.legend()
    plt.xlabel('time [d]')
    plt.ylabel('individuals')
    plt.show()
    #plt.savefig('Try_1.pdf')
