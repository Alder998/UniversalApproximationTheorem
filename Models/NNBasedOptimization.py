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

class normalPDFFunctionLayer(Layer):
    def __init__(self, **kwargs):
        super(normalPDFFunctionLayer, self).__init__(**kwargs)
        self.mu = self.add_weight(name='x', shape=(1,), initializer='random_normal', trainable=True)
        self.sigma = self.add_weight(name='y', shape=(1,), initializer='random_normal', trainable=True)

    def call(self, inputs):
        return d.Distributions(inputs).normalDistributionCDF(self.mu, self.sigma, operator='tf')

    def buildModel (self, X_train, Y_train, X_test, Y_test, epochs):

        # Define the Model
        inputs = Input(shape=(1,))
        outputs = normalPDFFunctionLayer()(inputs)
        model = Model(inputs, outputs)

        # Model Compilation
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Model Training
        model.fit(X_train, Y_train, epochs=epochs)

        # Test set Model evaluation
        loss = model.evaluate(X_test, Y_test, verbose=0)
        print(f'Test set Loss: {loss:.4f}')

        # Print Optimized Paramters
        optimized_weights = model.get_layer(index=1).get_weights()
        print(f'Optimized Parameters mu: {optimized_weights[0][0]}, sigma: {optimized_weights[1][0]}')

        # Predict the fitted values
        YPredictions = model.predict(X_test)

        return YPredictions







