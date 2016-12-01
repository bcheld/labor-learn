# this module initializes models and defines model functions
import numpy as np
import scipy

class Model(object):
    def __init__(self,
                 gamma_l = 0.2,     # share of lowest ability (permanently unemployed), mid, high
                 gamma_m = 0.6,     # share of lowest ability (permanently unemployed), mid, high
                 gamma_h = 0.2,     # share of lowest ability (permanently unemployed), mid, high
                 a = 1,             # prod fn constant
                 beta = 0.6,        # lower tier job share of output
                 r = 0.03,          # discount rate
                 ):
        # == enforce array == #
        params = gamma, a, beta, r
        params = map(np.asarray, params)
        gamma, a, beta, r = params

        self.gam_l = gamma_l
        self.gam_m = gamma_m
        self.gam_h = gamma_h
        self.a = a
        self.beta = beta
        self.r = r
        # convention on ordering of worker types within arrays: new, low, high, negative, positive
        self.p1 = np.array([
            self.gam_m + self.gam_h,                    # p1new
            1,                                          # p1low
            1,                                          # p1high
            self.gam_m / (self.gam_l + self.gam_m),     # p1neg
            1                                           # p1pos
        ])

        self.p2 = np.array([
            self.gam_h,                                 # p2new
            0,                                          # p2low
            1,                                          # p2high
            0,                                          # p2neg
            self.gam_h / (self.gam_h + self.gam_m),     # p2pos
        ])

    def prod(self,
             in1='blank',   # workers assigned to role 1
             in2='blank'    # workers assigned to role 2
             ):

        n1 = np.array([10, 10, 10, 10, 10]) if in1 == 'blank' else in1
        n2 = np.array([10, 10, 10, 10, 10]) if in2 == 'blank' else in2
        # role 1 expected input
        e_n1 = float(self.probs["p1new"]*n1[0] + self.probs["p1low"]*n1[1] + self.probs["p1high"]*n1[2]
                     + self.probs["p1neg"]*n1[3] + self.probs["p1pos"]+n1[4])
        # role 2 expected input
        e_n2 = float(self.probs["p2new"]*n2[0] + self.probs["p2low"]*n2[1] + self.probs["p2high"]*n2[2]
                     + self.probs["p2neg"]*n2[3] + self.probs["p2pos"]+n2[4])
        # role 1 expected input variance
        v_n1 = float(self.probs["p1new"] * (1-self.probs["p1new"]) * n1[0]
                     + self.probs["p1low"] * (1-self.probs["p1low"]) * n1[1]
                     + self.probs["p1high"] * (1-self.probs["p1high"]) * n1[2]
                     + self.probs["p1neg"] * (1-self.probs["p1neg"]) * n1[3]
                     + self.probs["p1pos"] * (1-self.probs["p1pos"]) * n1[4])
        # role 2 expected input variance
        v_n2 = float(self.probs["p2new"] * (1-self.probs["p2new"]) * n2[0]
                     + self.probs["p2low"] * (1-self.probs["p2low"]) * n2[1]
                     + self.probs["p2high"] * (1-self.probs["p2high"]) * n2[2]
                     + self.probs["p2neg"] * (1-self.probs["p2neg"]) * n2[3]
                     + self.probs["p2pos"] * (1-self.probs["p2pos"]) * n2[4])
        # output adjustment for role 1 uncertainty
        adj_1 = 1 - self.beta * (1-self.beta) * v_n1 / (2 * e_n1 ** 2)
        # output adjustment for role 2 uncertainty
        adj_2 = 1 - self.beta * (1-self.beta) * v_n2 / (2 * e_n2 ** 2)
        # return expected output
        return self.a * e_n1 ** self.beta * e_n2 ** (1-self.beta) * adj_1 * adj_2

    def wage_expense(self,
                     n1=(10, 10, 10, 10, 10), # workers assigned to role 1
                     n2=(10, 10, 10, 10, 10),  # workers assigned to role 2
                     w1=(10, 10, 10, 10, 10), # wages corresponding to role 1
                     w2=(10, 10, 10, 10, 10)  # wages corresponding to role 2
                     ):
        w1 = np.dot(n1, w1)
        w2 = np.dot(n2, w2)
        return w1 + w2

    r"""
    Need to define functions for production and profits given staffing
    Do something to solve for wages and staffing levels by firm type
    """
