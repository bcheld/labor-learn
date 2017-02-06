"""
Filename: model.py

Author: Brian Held

A class to define an instance of the learning model described in Ch. 2 of my thesis

"""
import numpy as np


class Model(object):
    """
    An instance of the class is an object with data on a particular
    problem of this type, including probabilities, discount factor and
    sample space for the variables.

    Parameters
    ----------
    gamma_l : population share of lowest ability (permanently unemployed)
    gamma_m : population share of mid ability (only produces in low job type)
    gamma_h : population share of high ability (produces in all job types)
    a       : production function constant
    beta    : lower tier job share of output
    r       : discount rate

    Attributes
    ----------
    gamma_l, gamma_m, gamma_h, a, beta, r : see Parameters
    p1 : array_like(float, ndim=1)
        The probabilities of success in the low job type for all worker types
    p2 : array_like(float, ndim=1)
        The probabilities of success in the high job type for all worker types
    v1_inpt : array_like(float, ndim=1)
        P(1-P) for all workers types in the low job type (used for variance calculations in methods)
    v2_inpt : array_like(float, ndim=1)
        P(1-P) for all workers types in the high job type (used for variance calculations in methods)


    Worker Type Indexing
    --------------------
    0 : new - new worker
    1 : low - worker with known ability gam_m
    2 : high - worker with known ability gam_h
    3 : neg - worker known not to be gam_h
    4 : pos - worker known to be at least gam_m

    In combined arrays (worker type-specific values corresponding to both low and high types), low
    is first (indexed [0:4]) and high is last (indexed [5:9])
    """

    def __init__(self, gamma_l=0.2, gamma_m=0.6, gamma_h=0.2, a=1, beta=0.6, r=0.03):
        self.gam_l, self.gam_m, self.gam_h = gamma_l, gamma_m, gamma_h
        self.a, self.beta, self.r = a, beta, r

        self.p1 = np.array([self.gam_m + self.gam_h, 1, 1, self.gam_m / (self.gam_l + self.gam_m), 1])
        self.p2 = np.array([self.gam_h, 0, 1, 0, self.gam_h / (self.gam_h + self.gam_m)])

        self.v1_inpt = self.p1 * (1 - self.p1)
        self.v2_inpt = self.p2 * (1 - self.p2)

    def prod(self, n1new, n1low, n1high, n1neg, n1pos, n2new, n2low, n2high, n2neg, n2pos):
        """
        Production as a function of labor input by type/assignment

        Parameters
        ----------
        n1new  : new type workers assigned to low job type
        n1low  : low type workers assigned to low job type
        n1high : high type workers assigned to low job type
        n1neg  : neg type workers assigned to low job type
        n1pos  : pos type workers assigned to low job type
        n2new  : new type workers assigned to high job type
        n2low  : low type workers assigned to high job type
        n2high : high type workers assigned to high job type
        n2neg  : neg type workers assigned to high job type
        n2pos  : pos type workers assigned to high job type

        Returns
        -------
        production : array_like(float)
            The expected output of the firm given the worker allocation

        """
        n1 = np.array([n1new, n1low, n1high, n1neg, n1pos])
        n2 = np.array([n2new, n2low, n2high, n2neg, n2pos])

        # expected input for role 1 and 2
        e_n1 = self.p1 @ n1
        e_n2 = self.p2 @ n2

        # expected input variance for roles 1 and 2
        v_n1 = self.v1_inpt @ n1
        v_n2 = self.v2_inpt @ n2

        # output adjustments for uncertainty in labor input by role
        adj_1 = 1 - self.beta * (1-self.beta) * v_n1 / (2 * e_n1 ** 2)
        adj_2 = 1 - self.beta * (1-self.beta) * v_n2 / (2 * e_n2 ** 2)

        # return expected output
        production = self.a * e_n1 ** self.beta * e_n2 ** (1-self.beta) * adj_1 * adj_2
        return production

    def allocate(self, n_new, n_low, n_high, n_neg, n_pos):
        """
        Method to optimally allocate a given set of workers within a single firm

        Parameters
        ----------
        n_new  : total new type workers
        n_low  : total low type workers
        n_high : total high type workers
        n_neg  : total neg type workers
        n_pos  : total pos type workers

        Returns
        -------
        alloc : array_like(int)
            Returns the optimal assignment of workers (length 10)

        Algorithm
        ---------
        1. start with low and neg type workers if any (can only work low job)
        2. fill high to start
        3. calculate gain from adding new to low role vs. adding pos to high role and keep highest, iterate until at
           least one of the roles are completely allocated
        4. calculate gain from assigning remaining role to high or low, keep highest output and iterate until allocated
        5. if all pos type assigned to low fill from high if necessary

        Notes
        -----
        at most one worker type can work both roles in competitive eq.
        proof is tbd

        """
        # Algorithm step 1&2
        alloc = [0, n_low, 0, n_neg, 0, 0, 0, n_high, 0, 0]
        # Algorithm step 3
        while alloc[0] + alloc[5] < n_new and alloc[4] + alloc[9] < n_pos:
            low = np.copy(alloc)
            low[0] += 1
            high = np.copy(alloc)
            high[9] += 1
            if self.prod(*low) >= self.prod(*high):
                alloc = np.copy(low)
            else:
                alloc = np.copy(high)
        # Algorithm step 4a - excess new type
        while alloc[0] + alloc[5] < n_new:
            low = np.copy(alloc)
            low[0] += 1
            high = np.copy(alloc)
            high[5] += 1
            if self.prod(*low) >= self.prod(*high):
                alloc = np.copy(low)
            else:
                alloc = np.copy(high)
        # Algorithm step 4b - excess pos type
        while alloc[4] + alloc[9] < n_new:
            low = np.copy(alloc)
            low[4] += 1
            high = np.copy(alloc)
            high[9] += 1
            if self.prod(*low) >= self.prod(*high):
                alloc = np.copy(low)
            else:
                alloc = np.copy(high)
        # Algorithm step 5 - check high type allocation
        low = np.copy(alloc)
        low[2] = 1
        low[7] -= 1
        while self.prod(*low) > self.prod(*alloc):
            alloc = np.copy(low)
            low[2] += 1
            low[7] -= 1
        return alloc


    """
    Need to define functions for production and profits given staffing
    Do something to solve for wages and staffing levels by firm type
    """

    """
    solves for optimal allocation of workers within a firm given staff
    1. start with low and neg type workers if any (can only work low job)
    2. fill high to start
    3. calculate gain from adding new to low vs. adding pos to high
    4. iterate until new and pos type workers assigned
    5. if all pos type assigned to low fill from high

    note: at most one worker type can work both roles in competitive eq.
    proof: tbd
    """