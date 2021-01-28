import epipack
import numpy as np
from bfmplot import pl as plt

class ParamDict(dict):
    """

    """
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self

class first_generation_tracing():
    """
    This class provides a first generation tracing system on an SIR-based pandemic.
    Parameter
    ---------
    Number of individuals (float) : N
    Quarantining of susceptible contacts (boolean): quarantine_S_contacts

    Example
    --------
    model = first_generation_tracing( N = 1000, quarantine_S_contacts = True)
    model.set_parameters(parameters)
    result = model.compute(t)

    """
    def __init__(self,N,quarantine_S_contacts):

        self.N = N
        self.quarantine_S_contacts = quarantine_S_contacts
        if self.quarantine_S_contacts == True:
            model = epipack.EpiModel(['S','I','R','X','Q'],initial_population_size = N)
            self.model = model
        else:
            model = epipack.EpiModel(['S','I','R','X'],initial_population_size = N)
            self.model = model

    def compute(self, t):
        """
        This function calculates the integration of the chosen model.
        Parameter
        ---------
        time (array or list) : t

        Returns
        -------
        Dictionary of results

        result = {
                'S': ....,
                'I': ....,
                }
        """
        result = self.model.integrate(t, integrator='dopri5')
        self.result = result
        return result

    def set_parameters(self,parameters):
        """
        This function sets the processes and parameters of the chosen model.
        Parameter
        ---------
        Dictionary of parameters (dict) : parameters
        parameters = {
                        'R0' : ...,
                        'I_0' : ...,
                        'recovery_rate' : ...,
                        'omega' : ..., # only needed if quarantine_S_contacts = True
                        'app_participation' : ...,
                        'q' : ...,
                        'number_of_contacts' : ...

        }
        """

        p = ParamDict(parameters)
        kappa = (p.q*p.recovery_rate)/(1-p.q)
        if self.quarantine_S_contacts == True:
            self.model.set_processes([
                        ('I', p.recovery_rate, 'R' ),
                        ('I', kappa, 'X' ),
                        ('Q', p.omega, 'S' ),
                        ('I', 'S', p.R0*p.recovery_rate, 'I', 'I' ),
                        ('I', 'S', kappa * p.app_participation**2 * p.number_of_contacts, 'I', 'Q' ),
                        ('I', 'I', kappa * p.app_participation**2 * p.number_of_contacts, 'I', 'X' )])

        else:
            self.model.set_processes([
                        ('I', p.recovery_rate, 'R' ),
                        ('I', kappa, 'X' ),
                        ('I', 'S', p.R0*p.recovery_rate, 'I', 'I' ),
                        ('I', 'I', kappa * p.app_participation**2 * p.number_of_contacts, 'I', 'X' )]),
        self.model.set_initial_conditions({'I': p.I_0, 'S': self.N - p.I_0})

class next_generation_tracing():
    """
    This class provides a next generation tracing system on an SIR-based pandemic.
    Parameter
    ---------
    Number of individuals (float) : N
    Quarantining of susceptible contacts (boolean): quarantine_S_contacts

    Example
    --------
    model = next_generation_tracing(N = 1000, quarantine_S_contacts = True)
    model.set_parameters(parameters)
    result = model.compute(t)

    """
    def __init__(self, N, quarantine_S_contacts):

        self.N = N
        self.quarantine_S_contacts = quarantine_S_contacts
        if self.quarantine_S_contacts == True:
            model = epipack.EpiModel(['S','I','R','T','X','Q'],initial_population_size = N)
            self.model = model
        else:
            model = epipack.EpiModel(['S','I','R','T','X'],initial_population_size = N)
            self.model = model

    def compute(self, t):
        """
        This function calculates the integration of the chosen model.
        Parameter
        ---------
        time (array or list) : t

        Returns
        -------
        Dictionary of results

        result = {
                'S': ....,
                'I': ....,
                }
        """
        result = self.model.integrate(t, integrator='dopri5')
        self.result = result
        return result

    def set_parameters(self,parameters):
        """
        This function sets the processes and parameters of the chosen model.
        Parameter
        ---------
        Dictionary of parameters (dict) : parameters
        parameters = {
                        'R0' : ...,
                        'I_0' : ...,
                        'recovery_rate' : ...,
                        'omega' : ..., # only needed if quarantine_S_contacts = True
                        'chi' : ...,
                        'app_participation' : ...,
                        'q' : ...,
                        'number_of_contacts' : ...

        }
        """
        p = ParamDict(parameters)
        kappa = (p.q*p.recovery_rate)/(1-p.q)
        if self.quarantine_S_contacts == True:
            self.model.set_processes([
                        ('I', 'S', p.R0*p.recovery_rate, 'I', 'I' ),
                        ('T', 'S', p.chi * p.app_participation**2 * p.number_of_contacts, 'T', 'Q' ),
                        ('T', 'I', p.chi * p.app_participation**2 * p.number_of_contacts, 'T', 'T' ),
                        ('I', p.recovery_rate, 'R'),
                        ('I', kappa, 'T'),
                        ('T', p.chi, 'X'),
                        ('Q', p.omega, 'S')])

        else:
            self.model.set_processes([
                                ('I', p.recovery_rate, 'R'),
                                ('I', kappa, 'T' ),
                                ('T', p.chi, 'X' ),
                                ('I', 'S', p.R0*p.recovery_rate, 'I', 'I'),
                                ('T', 'I', p.chi * p.app_participation**2 * p.number_of_contacts, 'T', 'T' )])
        self.model.set_initial_conditions({'I': p.I_0, 'S': self.N - p.I_0})



class mixed_tracing():
    """
    This class provides a mix of the first and next generation tracing system on an SIR-based pandemic.
    Parameter
    ---------
    Number of individuals (float) : N
    Quarantining of susceptible contacts (boolean): quarantine_S_contacts

    Example
    --------
    model = mixed_generation_tracing(N = 1000, quarantine_S_contacts = True)
    model.set_parameters(parameters)
    result = model.compute(t)

    """

    def __init__(self, N, quarantine_S_contacts):

        self.quarantine_S_contacts = quarantine_S_contacts
        self.N = N
        if self.quarantine_S_contacts == True:
            model = epipack.EpiModel(['S','E','I_P','I_S','I_A','R','T','X','Q'],initial_population_size = N)
            self.model = model
        else:
            model = epipack.EpiModel(['S','E','I_P','I_S','I_A','R','T','X'],initial_population_size = N)
            self.model = model

    def compute(self, t):
        """
        This function calculates the integration of the chosen model.
        Parameter
        ---------
        time (array or list) : t

        Returns
        -------
        Dictionary of results

        result = {
                'S': ....,
                'I': ....,
                }
        """
        result = self.model.integrate(t, integrator='dopri5')
        self.result = result
        return result

    def set_parameters(self,parameters):
        """
        This function sets the processes and parameters of the chosen model.
        Parameter
        ---------
        Dictionary of parameters (dict) : parameters
        parameters = {
                        'R0' : ...,
                        'I_0' : ...,
                        'beta' : ...,
                        'alpha' : ...,
                        'recovery_rate' : ...,
                        'omega' : ..., # only needed if quarantine_S_contacts = True
                        'chi' : ...,
                        'x' : ...,
                        'y' : ...,
                        'z' : ...,
                        'app_participation' : ...,
                        'q' : ...,
                        'number_of_contacts' : ...

        }
        """
        p = ParamDict(parameters)
        kappa = (p.q*p.recovery_rate)/(1-p.q)
        if self.quarantine_S_contacts == True:
            self.model.set_processes([
                        ('S','I_P',p.R0*p.beta/2,'E','I_P'),
                        ('S','I_A',p.R0*p.recovery_rate/4,'E','I_A'),
                        ('S','I_S',p.R0*p.recovery_rate/4,'E','I_S'),
                        ('E',p.alpha,'I_P'),
                        ('I_P',(1-p.x)*p.beta,'I_S'),
                        ('I_P',p.x*p.beta,'I_A'),
                        ('I_A',p.recovery_rate,'R'),
                        ('I_S',p.recovery_rate,'R'),
                        ('I_S',kappa,'T'),
                        ('T',p.chi,'X'),
                        ('E','T',p.chi*p.y*p.app_participation**2*p.number_of_contacts*p.z,'T','T'),
                        ('I_P','T',p.chi*p.y*p.app_participation**2*p.number_of_contacts*p.z,'T','T'),
                        ('I_S','T',p.chi*p.y*p.app_participation**2*p.number_of_contacts*p.z,'T','T'),
                        ('I_A','T',p.chi*p.y*p.app_participation**2*p.number_of_contacts*p.z,'T','T'),
                        ('S','T',p.chi*p.app_participation**2*p.number_of_contacts*p.z,'Q','T'),
                        ('E','T',p.chi*(1-p.y)*p.app_participation**2*p.number_of_contacts*p.z,'X','T'),
                        ('I_P','T',p.chi*(1-p.y)*p.app_participation**2*p.number_of_contacts*p.z,'X','T'),
                        ('I_S','T',p.chi*(1-p.y)*p.app_participation**2*p.number_of_contacts*p.z,'X','T'),
                        ('I_A','T',p.chi*(1-p.y)*p.app_participation**2*p.number_of_contacts*p.z,'X','T'),
                        ('Q',p.omega,'S')])
        else:
            self.model.set_processes([
                        ('S','I_P',p.R0*p.beta/2,'E','I_P'),
                        ('S','I_A',p.R0*p.recovery_rate/4,'E','I_A'),
                        ('S','I_S',p.R0*p.recovery_rate/4,'E','I_S'),
                        ('E',p.alpha,'I_P'),
                        ('I_P',(1-p.x)*p.beta,'I_S'),
                        ('I_P',p.x*p.beta,'I_A'),
                        ('I_A',p.recovery_rate,'R'),
                        ('I_S',p.recovery_rate,'R'),
                        ('I_S',kappa,'T'),
                        ('T',p.chi,'X'),
                        ('E','T',p.chi*p.y*p.app_participation**2*p.number_of_contacts*p.z,'T','T'),
                        ('I_P','T',p.chi*p.y*p.app_participation**2*p.number_of_contacts*p.z,'T','T'),
                        ('I_S','T',p.chi*p.y*p.app_participation**2*p.number_of_contacts*p.z,'T','T'),
                        ('I_A','T',p.chi*p.y*p.app_participation**2*p.number_of_contacts*p.z,'T','T'),
                        ('E','T',p.chi*(1-p.y)*p.app_participation**2*p.number_of_contacts*p.z,'X','T'),
                        ('I_P','T',p.chi*(1-p.y)*p.app_participation**2*p.number_of_contacts*p.z,'X','T'),
                        ('I_S','T',p.chi*(1-p.y)*p.app_participation**2*p.number_of_contacts*p.z,'X','T'),
                        ('I_A','T',p.chi*(1-p.y)*p.app_participation**2*p.number_of_contacts*p.z,'X','T')])

        self.model.set_initial_conditions({'I_P': p.I_0, 'S': self.N - p.I_0})


if __name__=="__main__":

    population_size = 80e6
    t = np.linspace(0,365,1000)
    model = first_generation_tracing(N = population_size, quarantine_S_contacts = True)
    parameter = {
            'R0': 2.5,
            'q': 0.3,
            'app_participation': 0.3,
            'chi':1/2.5,
            'recovery_rate' : 1/6,
            'alpha' : 1/2,
            'beta' : 1/2,
            'number_of_contacts' : 6.3,
            'x':0.4,
            'y':0.1,
            'z':0.64,
            'I_0' : 1000,
            'omega':1/10
            }
    model.set_parameters(parameter)
    result = model.compute(t)
    plt.plot(t,result['S'],label = 'S')
    #plt.plot(t,result['E'],label = 'E')
    #plt.plot(t,result['I_P'],label = 'I_P')
    #plt.plot(t,result['I_S'],label = 'I_S')
    #plt.plot(t,result['I_A'],label = 'I_A')
    plt.plot(t,result['R'],label = 'R')
    #plt.plot(t,result['T'],label = 'T')
    plt.plot(t,result['X'],label = 'X')
    plt.plot(t,result['Q'],label = 'Q')
    plt.legend()
    plt.xlabel('time [d]')
    plt.ylabel('individuals')
    plt.show()
