import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class utils:

    def __init__ (self):
        pass

    class functions:

        def __init__ (self):
            pass

        def generateFunctionByPoints (self, function, pointStart=0, pointEnd=10, steps=500):

            pointsList = list()
            xSpace = np.linspace(pointStart, pointEnd, steps)
            for value in xSpace:
                pointsList.append(function(value))

            return [xSpace, pointsList]

        def plotFunction (self, pointsList):

            functionSet = pointsList

            plt.figure(figsize=(15, 5))
            plt.plot(functionSet)
            plt.show()

    class dataSetPreparation:

        def __init__ (self, yAxis, xAxis):
            self.yAxis = yAxis
            self.xAxis = xAxis
            pass

        def trainTestSplit (self, test_size = 0.25, random_state=1893):

            # Split the datatset into Train and Test, for the both variables

            XTrain, XTest, YTrain, YTest = train_test_split(self.xAxis, self.yAxis,
                                               random_state = random_state,
                                               test_size = test_size)

            TrainX = np.array(XTrain)
            TrainY = np.array(YTrain)
            TestX = np.array(XTest)
            TestY = np.array(YTest)

            return [TrainX, TrainY, TestX, TestY]


