import math
import numpy as np
import matplotlib.pyplot as plt
from BasicFunctions import Functions

class utils:

    def __init__ (self):
        pass

    class functions:

        def __init__ (self, function):
            self.function = Functions
            pass

        def generateFunctionByPoints (self, pointStart=0, pointEnd=10, steps=500):

            pointsList = list()
            for value in np.linspace(pointStart, pointEnd, steps):
                pointsList.append(self.function.functions(value))

            return pointsList

        def plotFunction (self):

            functionSet = self.generateFunctionByPoints()

            plt.figure(figsize=(15, 5))
            plt.plot(functionSet)
            plt.show()

