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

class LogisticDistribution:

    def __init__(self, rand, location, scale):
        self.rand = random
        self.location = location
        self.scale = scale

    def pdf(self, x):
        e_to_the_power = math.exp(-(x - self.location) / self.scale)
        pdf_value = e_to_the_power / (self.scale * (1 + e_to_the_power) ** 2)
        return pdf_value

    def cdf(self, x):
        return 0.5 + 0.5 * math.tanh((x - self.location) / (2 * self.scale))

    def ppf(self, p):
        return self.location + self.scale * math.log(p / (1 - p))

    def gen_rand(self):
        u = self.rand.random()
        return self.location + self.scale * math.log(u / (1 - u))

    def mean(self):
        return self.location

    def variance(self):
        return self.scale ** 2 * math.pi ** 2 / 3

    def skewness(self):
        return 0

    def ex_kurtosis(self):
        return 6/5

    def mvsk(self):
        return [self.mean(), self.variance(), 0, 6/5]

from scipy.special import gammainc, gamma, gammaincinv

class ChiSquaredDistribution:
    def __init__(self, rand, dof):
        if dof <= 0:
            raise ValueError("A szabadságfoknak pozitívnak kell lennie.")
        self.rand = random
        self.dof = dof

    def pdf(self, x):
        return x ** ((self.dof)/2 - 1) * math.exp(-x/2) / (2 ** (self.dof / 2) * scipy.special.gamma(self.dof / 2))

    def cdf(self, x):
        if x < 0:
            return 0.0

        return scipy.special.gammainc(self.dof / 2.0, x / 2.0)

    def ppf(self, p):
        if p < 0.0 or p > 1.0:
            raise ValueError("Az 'p' értéke 0 és 1 között kell legyen.")

        return 2.0 * scipy.special.gammaincinv(self.dof / 2.0, p)

    def gen_rand(self):
        u = self.rand.random()
        return self.ppf(u)

    def mean(self):
        return self.dof

    def variance(self):
        return self.dof * 2

    def skewness(self):
        return math.sqrt( 8 / self.dof)

    def ex_kurtosis(self):
        return 12 / self.dof

    def mvsk(self):
        return [self.mean(), self.variance(), self.skewness(), self.ex_kurtosis()]

