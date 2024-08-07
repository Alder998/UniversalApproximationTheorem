import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.math as tfm

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

        def gamma_approx(self, x):
            return tf.sqrt(2 * np.pi / x) * (x / np.e) ** x

        def regularized_beta(self, x, a, b):
            # a and b must be the same size
            a = tf.broadcast_to(a, tf.shape(x))
            b = tf.broadcast_to(b, tf.shape(x))
            return tfm.betainc(a, b, x)

        @tf.function
        def hyp2f1_series(self, a, b, c, z, terms=50):
            result = tf.ones_like(z)
            term = tf.ones_like(z)
            for n in range(1, terms):
                term *= (a + n - 1) * (b + n - 1) / ((c + n - 1) * n) * z
                result += term
            return result

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


