import numpy as np
import warnings
from scipy.integrate import odeint

class Lorenz(object):
    """
    Simulate the dynamics of the Lorenz equations
    """
    def __init__(self, sigma=10, rho=28, beta=2.667):
        """
        Inputs
        - sigma : float, the Prandtl number
        - rho : float, the Rayleigh number
        - beta : float, the spatial scale
        """
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
    
    def __call__(self, X, t):
        """
        The dynamical equation for the Lorenz system
        - X : tuple of length 3, corresponding to the three coordinates
        - t : float (the current time)
        """
        
        x, y, z = X
        
        xdot = self.sigma*(y - x)
        ydot = x*(self.rho - z) - y
        zdot = x*y - self.beta*z
        return (xdot, ydot, zdot)
    
    def integrate(self, X0, tpts):
        """
        X0 : 3-tuple, the initial values of the three coordinates
        tpts : np.array, the time mesh
        """
        x0, y0, z0 = X0
        sol = odeint(self, (x0, y0, z0), tpts)
        return sol.T
    
    
class MacArthur(object):
    """
    Simulate the dynamics of the modified MacArthur resource competition model,
    as studied by Huisman & Weissing, Nature 1999
    """
    def __init__(self, r=None, k=None, c=None, d=None, s=None, m=None):
        """
        Inputs
        """
        
        if r==None:
            self.set_defaults()
        else:
            assert len(s) == k.shape[0], "vector \'s\' has improper dimensionality"
            assert k.shape == c.shape, "K and C matrices must have matching dimensions"
            self.r = r
            self.k = k
            self.c = c
            self.d = d
            self.s = s
            self.m = m
            
        self.n_resources, self.n_species = self.k.shape
                
    def set_defaults(self):
        """
        Set default values for parameters. Taken from Fig. 4 of 
        Huisman & Weissing. Nature 1999
        """
        
        self.k = np.array([[0.39,0.34,0.30,0.24,0.23,0.41,0.20,0.45,0.14,0.15,0.38,0.28],
                           [0.22,0.39,0.34,0.30,0.27,0.16,0.15,0.05,0.38,0.29,0.37,0.31],
                           [0.27,0.22,0.39,0.34,0.30,0.07,0.11,0.05,0.38,0.41,0.24,0.25],
                           [0.30,0.24,0.22,0.39,0.34,0.28,0.12,0.13,0.27,0.33,0.04,0.41],
                           [0.34,0.30,0.22,0.20,0.39,0.40,0.50,0.26,0.12,0.29,0.09,0.16]])
        self.c = np.array([[0.04,0.04,0.07,0.04,0.04,0.22,0.10,0.08,0.02,0.17,0.25,0.03],
                           [0.08,0.08,0.08,0.10,0.08,0.14,0.22,0.04,0.18,0.06,0.20,0.04],
                           [0.10,0.10,0.10,0.10,0.14,0.22,0.24,0.12,0.03,0.24,0.17,0.01],
                           [0.05,0.03,0.03,0.03,0.03,0.09,0.07,0.06,0.03,0.03,0.11,0.05],
                           [0.07,0.09,0.07,0.07,0.07,0.05,0.24,0.05,0.08,0.10,0.02,0.04]])
        self.s = np.array([6, 10, 14, 4, 9])
        self.d = 0.25
        self.r = 1
        self.m = 0.25
        
        # 5 species, 5 resources
        self.k = self.k[:,:5]
        self.c = self.c[:,:5]
    
    def set_ic(self):
        """
        Get default initial conditions from Huisman & Weissing. Nature 1999
        """
        if self.n_species<=5:
            ic_n = np.array([0.1 + i/100 for i in range(1,self.n_species+1)])
        else:
            ic_n = np.hstack([np.array([0.1 + i/100 for i in range(1,5+1)]), np.zeros(n_species-5)])
        ic_r = np.copy(self.s)
        return (ic_n, ic_r)
    
    def growth_rate(self, rr):
        """
        Calculate growth rate using Liebig's law of the maximum
        r : np.ndarray, a vector of resource abundances
        """
        u0 = rr/(self.k.T + rr)
        u = self.r * u0.T
        return np.min(u.T, axis=1)
        
    def __call__(self, X, t):
        """
        The dynamical equation for the Lorenz system
        - X : vector of length n_species + n_resources, corresponding to all dynamic variables
        - t : float (the current time)
        """
        
        nn, rr = X[:self.n_species], X[self.n_species:]
        
        mu = self.growth_rate(rr)
        nndot = nn*(mu - self.m)
        rrdot = self.d*(self.s - rr) - np.matmul(self.c, (mu*nn))
        return np.hstack([nndot, rrdot])

    def integrate(self, X0, tpts):
        """
        X0 : 2-tuple of vectors, the initial values of the species and resources
        tpts : np.array, the time mesh
        """
        if not X0:
            X0 = self.set_ic()
        else:
            pass
        
        sol = odeint(self, np.hstack(X0), tpts)
        return sol.T

class Rossler(object):  
    """
    Simulate the dynamics of the Rossler attractor
    """
    def __init__(self, a=.2, b=.2, c=5.7):
        """
        Inputs
        - a : float
        - b : float
        - c : float
        """
        self.a = a
        self.b = b
        self.c = c
    
    def __call__(self, X, t):
        """
        The dynamical equation for the Lorenz system
        - X : tuple of length 3, corresponding to the three coordinates
        - t : float (the current time)
        """
        
        x, y, z = X
        
        xdot = -y - z
        ydot = x + self.a*y
        zdot = self.b + z*(x - self.c)
        return (xdot, ydot, zdot)
    
    def integrate(self, X0, tpts):
        """
        X0 : 3-tuple, the initial values of the three coordinates
        tpts : np.array, the time mesh
        """
        x0, y0, z0 = X0
        sol = odeint(self, (x0, y0, z0), tpts)
        return sol.T

class Torus2(object):
    """
    Simulate a minimal quasiperiodic flow on a torus
    """
    def __init__(self, r=1.0, a=0.5, n=15.3):
        """
        - r : the toroid radius
        - a : the (smaller) cross sectional radius
        - n : the number of turns per turn. Any non-integer
                value produces a quasiperiodic toroid
        """
        self.r = r
        self.a = a
        self.n = n
    
    def __call__(self, X, t):
        """
        The dynamical equation for the Lorenz system
        - X : tuple of length 3, corresponding to the three coordinates
        - t : float (the current time)
        """
        
        x, y, z = X
        
        xdot = (-self.a*self.n*np.sin(self.n*t))*np.cos(t) - (self.r + self.a*np.cos(self.n*t))*np.sin(t)
        ydot = (-self.a*self.n*np.sin(self.n*t))*np.sin(t) + (self.r + self.a*np.cos(self.n*t))*np.cos(t)
        zdot = self.a*self.n*np.cos(self.n*t)
        return (xdot, ydot, zdot)
    
    def integrate(self, X0, tpts):
        """
        X0 : 3-tuple, the initial values of the three coordinates
        tpts : np.array, the time mesh
        """
        x0, y0, z0 = X0
        sol = odeint(self, (x0, y0, z0), tpts)
        return sol.T
