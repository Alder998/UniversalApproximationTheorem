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

class NNbasedOptimization (Layer):
    def __init__(self, **kwargs):
        super(NNbasedOptimization, self).__init__(**kwargs)
        self.parametersToOptimize = []
        pass

    # Let's try to use the function that we got from Distribution

    def generateFunctionLayer (self, x):
        for singleParam in self.parametersToOptimize:
            self.add_weight(name=singleParam, shape=(1,), initializer='random_normal', trainable=True)

    def knownFunctionApproximator (self):

        inputs = Input(shape=(1,))
        outputs = self.generateFunctionLayer()
        model = Model(inputs, outputs)

        # Compile the Model
        model.compile(optimizer=Adam(learning_rate=0.01), loss=MeanSquaredError())

        # See the optimized Parameters
        optimized_weights = model.get_layer(index=1).get_weights()
        print(f'Ottimizzati x: {optimized_weights[0][0]}, y: {optimized_weights[1][0]}')

    def buildModel(self):

