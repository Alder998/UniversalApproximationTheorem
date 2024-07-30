# Here we are using an approach that is slightly different from the ones used below: the Neural Network-based Optimization
# Approach.
# Usually, a Neural Network optimizes a function by optimizing the matrix of weights in each layer.
# Instead, here we need to Optimize the function paramters themselves, and use the NN architecture to do it

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from Distributions import Distributions as d
import numpy as np
from tensorflow.keras import losses

class functionLayer:
    def __init__(self, Layer, **kwargs):
        super(functionLayer, self).__init__(**kwargs)
        self.layer = Layer

    def buildModel (self, X_train, Y_train, X_test, Y_test, epochs, multiple = False):

        # Define the Model
        inputs = Input(shape=(1,))
        outputs = self.layer(inputs)
        model = Model(inputs, outputs)

        # Model Compilation
        model.compile(optimizer='adam', loss=losses.MeanAbsoluteError())

        # Model Training
        model.fit(X_train, Y_train, epochs=epochs)

        # Test set Model evaluation
        loss = model.evaluate(X_test, Y_test, verbose=0)
        print(f'Test set Loss: {loss:.4f}')

        # Print Optimized Paramters
        optimized_weights = model.get_layer(index=1).get_weights()
        if multiple:
            print(f'Optimized Parameters mus: {optimized_weights[0]}, sigmas: {optimized_weights[1]}')
        else:
            print(f'Optimized Parameters mu: {optimized_weights[0][0]}, sigma: {optimized_weights[1][0]}')

        # Predict the fitted values
        YPredictions = model.predict(X_test)

        return YPredictions







