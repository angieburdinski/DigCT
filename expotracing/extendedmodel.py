import epipack
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from numpy import random
class ParamDict(dict):
    """

    """
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self

class first_generation_tracing():
    """
    This class provides a first generation tracing system on an SIR-based pandemic.

    Example
    --------
    model = first_generation_tracing( N = 1000, quarantine_S_contacts = True)
    model.set_parameters(parameters)
    result = model.compute(t)

    """
    def __init__(self,N,quarantine_S_contacts):
        """
        __init__ intializes the model object.
        Parameter
        ---------
        Number of individuals (float) : N
        Quarantining of susceptible contacts (boolean): quarantine_S_contacts
        """
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
        The compute function calculates the integration of the chosen model with
        integrator = 'dopri5'.
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

        Example
        -------
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

    Example
    --------
    model = next_generation_tracing(N = 1000, quarantine_S_contacts = True)
    model.set_parameters(parameters)
    result = model.compute(t)

    """
    def __init__(self, N, quarantine_S_contacts):
        """
        __init__ intializes the model object.
        Parameter
        ---------
        Number of individuals (float) : N
        Quarantining of susceptible contacts (boolean): quarantine_S_contacts
        """
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
        This function calculates the integration of the chosen model with
        integrator = 'dopri5'.
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

        Example
        -------
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
    This class provides a mix of the first and next generation tracing system
    on an SIR-based pandemic. !! Attention !! This model is for analyzing the
    parameters of the La Gomera study. Due to the changed tracing term the usual
    number of contacts (real) is exchanged with number of contacts (measured) per
    app participation leading to a linear influence of app participation multiplied
    with measured number of contacts.

    Example
    --------
    model = mixed_tracing(N = 1000, quarantine_S_contacts = True)
    model.set_parameters(parameters)
    result = model.compute(t)

    """

    def __init__(self, N, quarantine_S_contacts):
        """
        __init__ intializes the model object.
        Parameter
        ---------
        Number of individuals (float) : N
        Quarantining of susceptible contacts (boolean): quarantine_S_contacts
        """
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
        The compute function calculates the integration of the chosen model with
        integrator = 'dopri5'.
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

        Example
        --------
        parameters = {
                        'R0' : ...,
                        'I_0' : ...,
                        'beta' : ..., # rate of becoming symptomatic or asymptomatic
                        'alpha' : ...,# rate of becoming presymptomatic
                        'recovery_rate' : ...,
                        'omega' : ..., # only needed if quarantine_S_contacts = True
                        'chi' : ..., # rate of entering infected-quarantined compartment from traced
                        'x' : ..., # fraction of individuals becoming symptomatic
                        'y' : 0.1, # fraction of individuals participating in next generation tracing
                        'z' : 0.64, # fraction of individuals inducing tracing
                        'app_participation' : ...,
                        'q' : ..., # fraction of detected and isolated symptomatic individuals
                        'number_of_contacts' : 6.3

        }
        """
        p = ParamDict(parameters)
        kappa = (p.q*p.recovery_rate)/(1-p.q)
        if self.quarantine_S_contacts == True:
            self.model.set_processes([
                        ('S','I_P',p.R0*p.beta/2,'E','I_P'),
                        ('S','I_A',p.R0*p.recovery_rate/2,'E','I_A'),
                        ('S','I_S',p.R0*p.recovery_rate/2,'E','I_S'),
                        ('E',p.alpha,'I_P'),
                        ('I_P',(1-p.x)*p.beta,'I_S'),
                        ('I_P',p.x*p.beta,'I_A'),
                        ('I_A',p.recovery_rate,'R'),
                        ('I_S',p.recovery_rate,'R'),
                        ('I_S',kappa,'T'),
                        ('T',p.chi,'X'),
                        ('E','T',p.chi*p.y*p.app_participation*p.number_of_contacts*p.z,'T','T'),
                        ('I_P','T',p.chi*p.y*p.app_participation*p.number_of_contacts*p.z,'T','T'),
                        ('I_S','T',p.chi*p.y*p.app_participation*p.number_of_contacts*p.z,'T','T'),
                        ('I_A','T',p.chi*p.y*p.app_participation*p.number_of_contacts*p.z,'T','T'),
                        ('S','T',p.chi*p.app_participation*p.number_of_contacts*p.z,'Q','T'),
                        ('E','T',p.chi*(1-p.y)*p.app_participation*p.number_of_contacts*p.z,'X','T'),
                        ('I_P','T',p.chi*(1-p.y)*p.app_participation*p.number_of_contacts*p.z,'X','T'),
                        ('I_S','T',p.chi*(1-p.y)*p.app_participation*p.number_of_contacts*p.z,'X','T'),
                        ('I_A','T',p.chi*(1-p.y)*p.app_participation*p.number_of_contacts*p.z,'X','T'),
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
                        ('E','T',p.chi*p.y*p.app_participation*p.number_of_contacts*p.z,'T','T'),
                        ('I_P','T',p.chi*p.y*p.app_participation*p.number_of_contacts*p.z,'T','T'),
                        ('I_S','T',p.chi*p.y*p.app_participation*p.number_of_contacts*p.z,'T','T'),
                        ('I_A','T',p.chi*p.y*p.app_participation*p.number_of_contacts*p.z,'T','T'),
                        ('E','T',p.chi*(1-p.y)*p.app_participation*p.number_of_contacts*p.z,'X','T'),
                        ('I_P','T',p.chi*(1-p.y)*p.app_participation*p.number_of_contacts*p.z,'X','T'),
                        ('I_S','T',p.chi*(1-p.y)*p.app_participation*p.number_of_contacts*p.z,'X','T'),
                        ('I_A','T',p.chi*(1-p.y)*p.app_participation*p.number_of_contacts*p.z,'X','T')])

        self.model.set_initial_conditions({'I_P': p.I_0, 'S': self.N - p.I_0})

class stoch_mixed_tracing():
    """
    This class provides a stochastic model mixed generation system
    on an SIR-based pandemic. !! Attention !! This model is for analyzing the
    parameters of the La Gomera study. Due to the changed tracing term the usual
    number of contacts (real) is exchanged with number of contacts (measured) per
    app participation leading to a linear influence of app participation multiplied
    with measured number of contacts.

    Example
    --------
    model = stoch_mixed_tracing(G, quarantine_S_contacts = True)
    model.set_parameters(parameters)
    t, result = model.compute(t)

    """

    def __init__(self, G ,quarantine_S_contacts):
        """
        __init__ intializes the model object.
        Parameter
        ---------
        Network (nx-object) : G
        Quarantining of susceptible contacts (boolean): quarantine_S_contacts
        """
        self.quarantine_S_contacts = quarantine_S_contacts
        self.G = G
        self.N = len(G.nodes())
        self.edge_weight_tuples = [ (e[0], e[1], 1.0) for e in G.edges() ]
        self.k_norm = 2*len(self.edge_weight_tuples) / self.N
        if self.quarantine_S_contacts == True:
            model = epipack.StochasticEpiModel(['S','E','I_P','I_S','I_A','R','T','X','Sa','Ea','I_Pa','I_Sa','I_Aa','Ra','Ta','Xa','Qa'],self.N, self.edge_weight_tuples,directed=False)
            self.model = model

        else:
            model = epipack.StochasticEpiModel(['S','E','I_P','I_S','I_A','R','T','X','Sa','Ea','I_Pa','I_Sa','I_Aa','Ra','Ta','Xa'],self.N,self.edge_weight_tuples,directed=False)
            self.model = model

    def compute(self, time):
        """
        The compute function calculates the integration of the chosen model with
        integrator = 'dopri5'.
        Parameter
        ---------
        time (int) : time

        Returns
        -------
        time (list),  results (dict)

        """
        t, result = self.model.simulate(time)
        self.result = result
        self.t = t
        return t, result

    def set_parameters(self,parameters):
        """
        This function sets the processes and parameters of the chosen model.
        Parameter
        ---------
        Dictionary of parameters (dict) : parameters

        Example
        --------
        parameters = {
                        'R0' : ...,
                        'I_0' : ...,
                        'beta' : ..., # rate of becoming symptomatic or asymptomatic
                        'alpha' : ...,# rate of becoming presymptomatic
                        'recovery_rate' : ...,
                        'omega' : ..., # only needed if quarantine_S_contacts = True
                        'chi' : ..., # rate of entering infected-quarantined compartment from traced
                        'x' : ..., # fraction of individuals becoming symptomatic
                        'y' : 0.1, # fraction of individuals participating in next generation tracing
                        'z' : 0.64, # fraction of individuals inducing tracing
                        'app_participation' : ...,
                        'q' : ..., # fraction of detected and isolated symptomatic individuals


        }
        """
        p = ParamDict(parameters)
        kappa = (p.q*p.recovery_rate)/(1-p.q)
        IP0 = int((1-p.app_participation)*p.I_0)
        IPa0 = p.I_0 - IP0

        Sa0 = int(random.binomial(self.N-p.I_0, p.app_participation, 1))
        S0 = int(self.N - p.I_0 - Sa0)

        if self.quarantine_S_contacts == True:

            self.model.set_conditional_link_transmission_processes({

            ("Ta", "->", "Xa") : [
                    ("Xa", "I_Pa", p.z*p.y, "Xa", "Ta" ),
                    ("Xa", "I_Sa", p.z*p.y, "Xa", "Ta" ),
                    ("Xa", "I_Aa", p.z*p.y, "Xa", "Ta" ),
                    ("Xa", "Ea", p.z*p.y, "Xa", "Ta" ),
                    ("Xa", "Sa", p.z, "Xa", "Qa" ),
                    ("Xa", "I_Pa", p.z*(1-p.y), "Xa", "Xa" ),
                    ("Xa", "I_Sa", p.z*(1-p.y), "Xa", "Xa" ),
                    ("Xa", "I_Aa", p.z*(1-p.y), "Xa", "Xa" ),
                    ("Xa", "Ea", p.z*(1-p.y), "Xa", "Xa" )]

                    })

            self.model.set_link_transmission_processes([

                        ('I_Pa','S',p.R0/self.k_norm*p.beta/2,'I_Pa','E'),
                        ('I_Aa','S',p.R0/self.k_norm*p.recovery_rate/2,'I_Aa','E'),
                        ('I_Sa','S',p.R0/self.k_norm*p.recovery_rate/2,'I_Sa','E'),

                        ('I_P','Sa',p.R0/self.k_norm*p.beta/2,'I_P','Ea'),
                        ('I_A','Sa',p.R0/self.k_norm*p.recovery_rate/2,'I_A','Ea'),
                        ('I_S','Sa',p.R0/self.k_norm*p.recovery_rate/2,'I_S','Ea'),

                        ('I_Pa','Sa',p.R0/self.k_norm*p.beta/2,'I_Pa','Ea'),
                        ('I_Aa','Sa',p.R0/self.k_norm*p.recovery_rate/2,'I_Aa','Ea'),
                        ('I_Sa','Sa',p.R0/self.k_norm*p.recovery_rate/2,'I_Sa','Ea'),

                        ('I_P','S',p.R0/self.k_norm*p.beta/2,'I_P','E'),
                        ('I_A','S',p.R0/self.k_norm*p.recovery_rate/2,'I_A','E'),
                        ('I_S','S',p.R0/self.k_norm*p.recovery_rate/2,'I_S','E')])

            self.model.set_node_transition_processes([
                        ('E',p.alpha,'I_P'),
                        ('I_P',(1-p.x)*p.beta,'I_S'),
                        ('I_P',p.x*p.beta,'I_A'),
                        ('I_A',p.recovery_rate,'R'),
                        ('I_S',p.recovery_rate,'R'),
                        ('I_S',kappa,'T'),
                        ('T',p.chi,'X'),
                        ('Qa',p.omega,'Sa'),
                        ('Ea',p.alpha,'I_Pa'),
                        ('I_Pa',(1-p.x)*p.beta,'I_Sa'),
                        ('I_Pa',p.x*p.beta,'I_Aa'),
                        ('I_Aa',p.recovery_rate,'Ra'),
                        ('I_Sa',p.recovery_rate,'Ra'),
                        ('I_Sa',kappa,'Ta'),
                        ('Ta',p.chi,'Xa')])

        else:
            self.model.set_conditional_link_transmission_processes({

            ("Ta", "->", "Xa") : [
                    ("Xa", "I_Pa", p.z*p.y, "Xa", "Ta" ),
                    ("Xa", "I_Sa", p.z*p.y, "Xa", "Ta" ),
                    ("Xa", "I_Aa", p.z*p.y, "Xa", "Ta" ),
                    ("Xa", "Ea", p.z*p.y, "Xa", "Ta" ),
                    ("Xa", "I_Pa", p.z*(1-p.y), "Xa", "Xa" ),
                    ("Xa", "I_Sa", p.z*(1-p.y), "Xa", "Xa" ),
                    ("Xa", "I_Aa", p.z*(1-p.y), "Xa", "Xa" ),
                    ("Xa", "Ea", p.z*(1-p.y), "Xa", "Xa" )]

                    })

            self.model.set_link_transmission_processes([

                        ('I_Pa','S',p.R0/self.k_norm*p.beta/2,'I_Pa','E'),
                        ('I_Aa','S',p.R0/self.k_norm*p.recovery_rate/2,'I_Aa','E'),
                        ('I_Sa','S',p.R0/self.k_norm*p.recovery_rate/2,'I_Sa','E'),

                        ('I_P','Sa',p.R0/self.k_norm*p.beta/2,'I_P','Ea'),
                        ('I_A','Sa',p.R0/self.k_norm*p.recovery_rate/2,'I_A','Ea'),
                        ('I_S','Sa',p.R0/self.k_norm*p.recovery_rate/2,'I_S','Ea'),

                        ('I_Pa','Sa',p.R0/self.k_norm*p.beta/2,'I_Pa','Ea'),
                        ('I_Aa','Sa',p.R0/self.k_norm*p.recovery_rate/2,'I_Aa','Ea'),
                        ('I_Sa','Sa',p.R0/self.k_norm*p.recovery_rate/2,'I_Sa','Ea'),

                        ('I_P','S',p.R0/self.k_norm*p.beta/2,'I_P','E'),
                        ('I_A','S',p.R0/self.k_norm*p.recovery_rate/2,'I_A','E'),
                        ('I_S','S',p.R0/self.k_norm*p.recovery_rate/2,'I_S','E')])

            self.model.set_node_transition_processes([
                        ('E',p.alpha,'I_P'),
                        ('I_P',(1-p.x)*p.beta,'I_S'),
                        ('I_P',p.x*p.beta,'I_A'),
                        ('I_A',p.recovery_rate,'R'),
                        ('I_S',p.recovery_rate,'R'),
                        ('I_S',kappa,'T'),
                        ('T',p.chi,'X'),
                        ('Ea',p.alpha,'I_Pa'),
                        ('I_Pa',(1-p.x)*p.beta,'I_Sa'),
                        ('I_Pa',p.x*p.beta,'I_Aa'),
                        ('I_Aa',p.recovery_rate,'Ra'),
                        ('I_Sa',p.recovery_rate,'Ra'),
                        ('I_Sa',kappa,'Ta'),
                        ('Ta',p.chi,'Xa')])
        self.model.set_network(self.N,self.edge_weight_tuples)
        self.model.set_random_initial_conditions({ 'Sa' : Sa0,'S' : S0,'I_P':IP0,'I_Pa':IPa0})



if __name__=="__main__":
    parameter = {
            'R0': 2.5,
            'q': 0.3,
            'app_participation': 0.3,
            'chi':1/2.5,
            'recovery_rate' : 1/6,
            'alpha' : 1/2,
            'beta' : 1/2,
            'number_of_contacts' : 6.3,
            'x':0.17,
            'y':0.1,
            'z':0.64,
            'I_0' : 10,
            'omega':1/10
            }
    N = 100
    k0 = 50
    G = nx.barabasi_albert_graph(N,k0)
    model = stoch_mixed_tracing(G,False)
    model.set_parameters(parameter)
    t,result = model.compute(10000)
    print(t,result)
    plt.figure()

    for comp, series in result.items():
        plt.plot(t, series, label=comp)
    plt.legend()
    plt.show()
