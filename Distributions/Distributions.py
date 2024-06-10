import math
import numpy as np
import matplotlib.pyplot as plt

# It is not possible, as it is common belief, to approximate a distribution as
# it will be a generic function (i.e. generating random points, and making our net "Following" them). We can try to
# find alternative paths for our Net

class Distributions:
    def __init__ (self, x=0):
        self.x = x
        pass

    # Now the normal PDF must be implemented: it is: 1/(sqrt(2pi * sigma**2)) * e^((-1/2)*((x - mu) / sigma)**2)
    def normalDistributionPDF(self, sigma=1, mu=0):
        closedForm = 1 / (np.sqrt(2*np.pi * sigma ** 2)) * np.exp((-1 / 2) * ((self.x - mu) / sigma) ** 2)
        return closedForm