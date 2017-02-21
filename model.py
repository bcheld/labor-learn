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
    beta    : low type role share of output
    r       : discount rate
    learn   : arrival rate of ability signals

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

    def __init__(self, gamma_l=0.2, gamma_m=0.6, gamma_h=0.2, a=1, beta=0.6, r=0.03, learn=0.1):
        self.gam_l, self.gam_m, self.gam_h = gamma_l, gamma_m, gamma_h
        self.a, self.beta, self.r, self.learn = a, beta, r, learn

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
        prod : array_like(float)
            Returns the output associated with optimal assignment

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
        if n_low + n_neg == 0:
            if n_new > 0:
                alloc[0] += 1
            else:
                alloc[4] += 1
        if n_high == 0:
            if n_pos > 0:
                alloc[9] += 1
            else:
                alloc[5] += 1
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
        while alloc[4] + alloc[9] < n_pos:
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
        return alloc, self.prod(*alloc)

    def simt(self, n_new, n_low, n_high, n_neg, n_pos, time=0):
        """
        Method to simulate worker transitions and output over t periods starting with an initial worker state

        Parameters
        ----------
        time   : number of periods to simulate (simulate until convergence if t=0)
        n_new  : total new type workers at t=0
        n_low  : total low type workers at t=0
        n_high : total high type workers at t=0
        n_neg  : total neg type workers at t=0
        n_pos  : total pos type workers at t=0

        Returns
        -------
        wkrs : array_like(int, t x 5)
            State vector of workers for each time period
        alloc : array_like(int, t x 10)
            Returns the optimal assignment of workers (length 10) for each time period
        prod : array_like(float, length t)
            Returns the output associated with optimal assignment for each time period

        Notes
        -----
        The uncertainty resolves at the aggregate level. There is opportunity to do firm level simulation
        in which realized output updates the firm-level beliefs about individual worker ability.

        """
        # Positive signal transition matrix - low job
        wintrans1 = np.array([[-1, 0, 0, 0, 1],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 1, 0, -1, 0],
                             [0, 0, 0, 0, 0]])
        # Positive signal transition matrix - high job
        wintrans2 = np.array([[-1, 0, 1, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 1, 0, -1]])
        # Negative signal transition matrix - low job
        losetrans1 = np.array([[-1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, -1, 0],
                             [0, 0, 0, 0, 0]])
        # Negative signal transition matrix - high job
        losetrans2 = np.array([[-1, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 1, 0, 0, -1]])
        death = 0.05
        wkrs = np.zeros([2,5], dtype=np.int32)
        alloc = np.zeros([2,10], dtype=np.int32)
        prod = np.zeros([2])
        unemployed = np.zeros([2], dtype=np.int32)
        wkrs[0] = np.array([n_new, n_low, n_high, n_neg, n_pos])
        alloc[0], prod[0] = self.allocate(*wkrs[0])
        t = 0
        flag = 0
        while flag == 0:
            t += 1
            if t > 1:
                wkrs = np.append(wkrs, np.zeros((1, 5), dtype=np.int32), axis=0)
                alloc = np.append(alloc, np.zeros((1, 10), dtype=np.int32), axis=0)
                prod = np.append(prod, np.zeros(1), axis=0)
                unemployed = np.append(unemployed, np.zeros(1, dtype=np.int32), axis=0)
            # resolve uncertainty
            learned = np.random.binomial(alloc[t-1], self.learn, size=10)
            win1 = np.random.binomial(learned[0:5], self.p1, size=5)
            chg1 = win1 @ wintrans1 + (learned[0:5]-win1) @ losetrans1
            win2 = np.random.binomial(learned[5:10], self.p2, size=5)
            chg2 = win2 @ wintrans2 + (learned[5:10]-win2) @ losetrans2
            wkrs[t] = chg1 + chg2 + wkrs[t-1]
            dead = np.random.binomial(wkrs[t], death, size=5)
            dead_unemployed = np.random.binomial(unemployed[t-1], death)
            unemployed[t] = unemployed[t-1] - np.sum((learned[0:5]-win1) @ losetrans1) - dead_unemployed
            wkrs[t] -= dead
            wkrs[t, 0] += np.sum(dead) + dead_unemployed
            alloc[t], prod[t] = self.allocate(*wkrs[t])
            if t == time or (t > time and sum(np.subtract(wkrs[t], wkrs[t-1])) <= max(5, sum(wkrs[t]) * 0.05)):
                flag = 1
        return wkrs, alloc, prod
