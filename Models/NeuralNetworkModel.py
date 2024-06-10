# This module is dedicated to the core of Universal Approximation Theorem, i.e., the Neural Network Model that is core to
# Approximate the continuous functions

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
class NNModel:

    def __init__ (self):
        pass

    def trainAndEvaluateNNModelForBasicFunctions (self, XTrain, YTrain, XTest, YTest, epochs = 50):

        model = keras.Sequential ([
            layers.Dense(200, activation = 'relu', input_shape=(1,)),
            layers.Dense(1)
        ])
        # Compile and fit the Model
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(XTrain, YTrain, epochs=epochs)

        # Evaluate the performance of the Model
        loss = model.evaluate(XTest, YTest)
        print('Train Process ended, Loss:', loss)

        # Predict the fitted values
        YPredictions = model.predict(XTest)

        return YPredictions

