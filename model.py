# this module initializes models and defines model functions

class Model(object):
    def __init__(self,
                 gaml,  # share of lowest ability (permanently unemployed)
                 gamm,  # share of mid ability
                 gamh,  # share of high ability
                 a,     # prod fn constant
                 beta,  # lower tier job share of output
                 r,     # discount rate
                 ):
        assert gaml + gamm + gamh == 1
        self.gam = (gaml, gamm, gamh)
        self.a = a
        self.beta = beta
        self.r = r
        self.probs = {}
        self.probs["p1new"] = self.gam[1] + self.gam[2]
        self.probs["p1low"] = 1
        self.probs["p1high"] = 1
        self.probs["p1neg"] = self.gam[1] / (self.gam[0] + self.gam[1])
        self.probs["p1pos"] = 1
        self.probs["p2new"] = self.gam[2]
        self.probs["p2low"] = 0
        self.probs["p2high"] = 1
        self.probs["p2neg"] = 0
        self.probs["p2pos"] = self.gam[2] / (self.gam[1] + self.gam[2])

    def prod(self, n1=(10, 10, 10, 10, 10), n2=(10, 10, 10, 10, 10)):
        e_n1 = (self.probs["p1new"]*n1[0] + self.probs["p1low"]*n1[1] + probs["p1high"]*n1[2] +
            probs["p1neg"]*n1[3] + probs["p1pos"]+n1[4])
        e_n2 = (self.probs["p2new"] * n2[0] + self.probs["p2low"] * n2[1] + probs["p2high"] * n2[2] +
            probs["p2neg"] * n2[3] + probs["p2pos"] + n2[4])

