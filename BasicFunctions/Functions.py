
import math
import numpy as np
import matplotlib.pyplot as plt

class Functions:
    def __init__ (self, x=0):
        self.x = x
        pass

    # Let's build simple continuous function

    # Sine function
    def sineFunction(self):
        return math.sin(self.x)

    # Cosine function
    def cosineFunction(self):
        return math.cos(self.x)

    # Exponential Function
    def exponentialFunction(self):
        return math.exp(self.x)

    # Quadratic function
    def quadraticFunction (self):
        return self.x ** 2

