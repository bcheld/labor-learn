# this program defines the firm class according to Ch. 2

class Firm(object):
    def __init__(self, n1, n2):
        self.n1low, self.n1high, self.n1neg, self.n1pos, self.n1new = n1
        self.n2low, self.n2high, self.n2neg, self.n2pos, self.n2new = n2

    def __str__(self):
        return print('N1 is allocated as:', self.n1new, self.n1low, self.n1high, self.n1neg, self.n1pos,
                     '\nN2 is allocated as:', self.n2new, self.n2low, self.n2high, self.n2neg, self.n2pos, sep=' ')

