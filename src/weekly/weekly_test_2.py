import math
import random
import typing


class LaplaceDistribution:

    def __init__(self, rand, loc, scale):
        self.rand = rand
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        return 1 / (2 * self.scale) * math.exp(- (abs(x - self.loc)) / self.scale)

    def cdf(self, x):
        def sgn(x):
            if x < 0:
                return -1
            elif x == 0:
                return 0
            else:
                return 1

        return 1 / 2 + 1 / 2 * sgn(x - self.loc) * (1 - math.exp(- (abs(x - self.loc)) / self.scale))

    def ppf(self, p):
        def sgn(x):
            if x < 0:
                return -1
            elif x == 0:
                return 0
            else:
                return 1
        return self.loc - self.scale * sgn(p - 0.5) * math.log(1 - 2 * abs(p - 0.5))

    def gen_rand(self):
        u = random.uniform(-0.5, 0.5)
        def sgn(x):
            if x < 0:
                return -1
            elif x == 0:
                return 0
            else:
                return 1
        return self.loc - self.scale * sgn(u) * math.log(1 - 2 * abs(u))

    def mean(self):
        return self.loc

    def variance(self):
        return 2 * (self.scale ** 2)

    def skewness(self):
        return 0

    def ex_kurtosis(self):
        return 3

    def mvsk(self):
        return [self.loc, self.variance(), self.skewness(), self.ex_kurtosis()]


class ParetoDistribution:
    def __init__(self, rand, scale, shape):
        self.rand = rand
        self.scale = scale
        self.shape = shape

    def pdf(self, x):
        if x >= self.scale:
            return (self.shape * (self.scale ** self.shape)) / (x ** (self.shape + 1))

        else:
            return 0

    def cdf(self, x):

        if x >= self.scale:
            return 1 - (self.scale / x) ** self.shape

        else:
            return 0

    def ppf(self, p):
        if p > 0 and p < 1:
            return self.scale / (1 - p) ** (1 / self.shape)
        else:
            raise ValueError("p must be in the range (0, 1)")

    def gen_rand(self):
        u = random.uniform(0, 1)
        return self.scale / ((1 - u) ** (1 / self.shape))

    def mean(self):
        if self.shape <= 1:
            raise Exception("Moment undefined")
        else:
            return (self.shape * self.scale) / (self.shape - 1)

    def variance(self):
        if self.shape <= 2:
            raise Exception("Moment undefined")
        else:
            return ((self.scale ** 2) * self.shape) /((self.shape - 1) ** 2 * (self.shape - 2))

    def skewness(self):
        if self.shape <= 3:
            raise Exception("Moment undefined")
        else:
            return 2 * (1 + self.shape) / (self.shape - 3) * math.sqrt((self.shape - 2) / self.shape)

    def ex_kurtosis(self):
        if self.shape <= 4:
            raise Exception("Moment undefined")
        else:
            return 6 * (self.shape ** 3 + self.shape ** 2 - 6 * self.shape - 2) / (self.shape * (self.shape - 3) * (self.shape - 4))

    def mvsk(self):
        return [self.mean(), self.variance(), self.skewness(), self.ex_kurtosis()]