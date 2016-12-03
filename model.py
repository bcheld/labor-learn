# this module initializes models and defines model functions
import numpy as np


class Model(object):
    def __init__(self,
                 gamma_l=0.2,     # share of lowest ability (permanently unemployed), mid, high
                 gamma_m=0.6,     # share of lowest ability (permanently unemployed), mid, high
                 gamma_h=0.2,     # share of lowest ability (permanently unemployed), mid, high
                 a=1,             # prod fn constant
                 beta=0.6,        # lower tier job share of output
                 r=0.03,          # discount rate
                 ):
        self.gam_l = gamma_l
        self.gam_m = gamma_m
        self.gam_h = gamma_h
        self.a = a
        self.beta = beta
        self.r = r
        # Calculate probabilities for success in each role by type
        # convention on ordering of worker types within arrays: new, low, high, negative, positive (defined below)
        self.p1 = np.array([
            self.gam_m + self.gam_h,                    # p1new - new worker
            1,                                          # p1low - worker with known ability gam_m
            1,                                          # p1high - worker with known ability gam_h
            self.gam_m / (self.gam_l + self.gam_m),     # p1neg - worker known not to be gam_h
            1                                           # p1pos - worker known to be at least gam_m
        ])
        self.p2 = np.array([
            self.gam_h,                                 # p2new
            0,                                          # p2low
            1,                                          # p2high
            0,                                          # p2neg
            self.gam_h / (self.gam_h + self.gam_m)      # p2pos
        ])
        # Calculate p(1-p) for variance calculation later
        self.v1_inpt = self.p1 * (1 - self.p1)
        self.v2_inpt = self.p2 * (1 - self.p2)

    # expected production function
    def prod(self,
             n1new,   # new type workers assigned to role 1
             n1low,   # low type workers assigned to role 1
             n1high,  # high type workers assigned to role 1
             n1neg,   # neg type workers assigned to role 1
             n1pos,   # pos type workers assigned to role 1
             n2new,   # new type workers assigned to role 2
             n2low,   # low type workers assigned to role 2
             n2high,  # high type workers assigned to role 2
             n2neg,   # neg type workers assigned to role 2
             n2pos,   # pos type workers assigned to role 2
             ):
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
        return self.a * e_n1 ** self.beta * e_n2 ** (1-self.beta) * adj_1 * adj_2

    """
    Need to define functions for production and profits given staffing
    Do something to solve for wages and staffing levels by firm type
    """
