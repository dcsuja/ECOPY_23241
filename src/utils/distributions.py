import scipy.special


class FirstClass:
    pass


class SecondClass:
    def __init__(self, rand):
        self.rand = rand

class UniformDistribution:
    def __init__(self, rand, a, b):
        self.rand = rand
        self.a = a
        self.b = b

    def pdf(self, x):
        if self.a <= x <= self.b:
            return 1/(self.b - self.a)
        else:
           return 0

    def cdf(self, x):
        if x < self.a:
            return 0
        elif self.a <= x <= self.b:
            return (x - self.a) / (self.b - self.a)
        else:
            return 1

    def ppf(self, p):
        if 0 < p < 1:
             return self.a + p * (self.b - self.a)

    def gen_random(self):
        return self.a + (self.b - self.a) * self.rand

    def mean(self):
        if (self.a + self.b)/2 == 0:
            raise Exception("Moment undefined")
        else:
            return (self.a + self.b)/2

    def median(self):
        return (self.a + self.b)/2

    def variance(self):
        if ((self.b - self.a) ** 2) / 12 == 0:
            raise Exception("Moment undefined")
        else:
            return ((self.b - self.a) ** 2) / 12

    def skewness(self):
        return 0

    def ex_kurtosis(self):
        return -6/5


    def mvsk(self):
        mean = self.mean()
        variance = self.variance()
        skewness = 0
        kurtosis = self.ex_kurtosis()

        return [mean, variance, skewness, kurtosis]


class NormalDistribution:
    def __init__(self, rand, loc, scale):
        self.rand = rand
        self.loc = loc
        self.scale = scale



    def pdf(self, x):
        import math
        probability_density = (math.exp(-1/2 * (( x - self.mean / self.scale ** 1/2) ** 2))) / ((self.scale * 2 * math.pi) ** 1/2)
        return probability_density

    def f(x):
        return math.exp((-x ** 2) / 2)
    def cdf(x):
        from scipy.integrate import quad
        negative_inf = -math.inf
        cumulative_distribution = quad(f, negative_inf, x) / ((2 * math.pi) ** 1/2)
        return cumulative_distribution

    from scipy.stats import norm
    def ppf(p):
        inverse_cdf = norm.ppf(p)

    import numpy as np
    rng = np.random.default_rng()
    def gen_random():
        random_number = rng.normal()
        return random_number

    def mean(self):
        return self.mean

    def median(self):
        return self.median

    def variance(self):
        return self.variance

    def skewness(self):
        return 0

    def ex_kurtosis(ex_kurtosis):
        return 0

    def mvsk(self):
        return [0, self.variance, 0, 0]


class CauchyDistribution:
    def __init__(self, rand, loc, scale):
        self.rand = rand
        self.loc = x0
        self.scale = gamma

    import math
    def pdf(self, x):
        probability_density = 1 / math.pi * ((self.scale / ((x - self.loc) ** 2) + self.scale ** 2 ))
        return probability_density

    def cdf(self, x):
        cumulative_distribution = (1 / math.pi) * math.atan((x - self.loc) / self.scale) + 1/2
        return cumulative_distribution

    def ppf(self, p):
        inverse_cumulative = self.loc + self.scale * math.tan(math.pi * (p - 1/2))

    def gen_random(self):
        random_number = math.tan(math.pi * (self.rand.random() - 1/2))
        return random_number

    def mean(self):
        raise Exception("Moment undefined")

    def median(self):
        return self.median

    def variance(self):
        raise Exception("Moment undefined")

    def skeweness(self):
        raise Exception("Moment undefined")

    def ex_kurtosis(self):
        raise Exception("Moment undefined")

    def mvsk(self):
        raise Exception("Moment undefined")


#ab9oac

import math
import random
import typing
from pyerf import pyerf

class LogisticDistribution:

    def __init__(self, rand, loc, scale):
        self.rand = rand
        self.loc = loc
        self.scale = scale

    def pdf(self, x):
        return (math.exp(-(x-self.loc)/self.scale)) / self.scale * (1 + math.exp(x-self.loc)/self.scale) ** 2)

    def cdf(self, x):
        return 0.5 + 0.5 * math.tanh((x - self.loc) / 2 * self.scale)

    def ppf(self, p):
        return self.loc + self.scale * math.log(p / (1 - p))

    def gen_rand(self):
        u = self.rand.random()
        return self.loc + self.scale * (1.0 - 2.0 * u) / (1.0 + 2.0 * abs(1.0 - 2.0 * u))

    def mean(self):
        return self.loc

    def variance(self):
        return self.scale ** 2 * math.pi ** 2 / 3

    def skeweness(self):
        return 0

    def ex_kurtosis(self):
        return 6/5

    def mvsk(self):
        return [self.mean(), self.variance(), 0, 6/5]

from scipy.special import gammainc, gamma, gammaincinv
class ChiSquaredDistribution:
    def __init__(self):
        self.rand = rand
        self.dof = dof

    def pdf(self, x):
        return  x ** ((self.dof)/2 - 1) * math.exp(-x/2) / (2 ** (self.dof / 2) * scipy.special.gamma(self.dof / 2))

    def cdf(self, x):
        return scipy.special.gamma(self.dof/2, x/2)

    def ppf(self, x):
        return (2 ** (-self.dof / 2)) / (scipy.special.gamma(self.dof / 2)) * x ** (-self.dof / 2 - 1) * math.exp ** (-1 / (2 * x))

    def gen_rand(self):
        cauchy_random = 0.0
        for _ in range(self.dof):
            cauchy_random += random.uniform(-1, 1)

        logistic_random = 1 / (1 + math.exp(-cauchy_random))

        return logistic_random

    def mean(self):
        return self.dof

    def variance(self):
        return self.dof * 2

    def skeweness(self):
        return math.sqrt( 8 / self.dof)

    def ex_kurtosis(self):
        return 12 / self.dof

    def mvsk(self):
        return [self.mean(), self.variance()]

