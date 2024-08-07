from tensorflow.keras.layers import Input, Layer
import tensorflow as tf

# Class created just to implement Layers for NN-based Optimization

class normalLayerFunctions (Layer):
    def __init__ (self, distributionParam, function, **kwargs):
        super(normalLayerFunctions, self).__init__(**kwargs)
        self.distributionParam = distributionParam
        self.function = function
        pass

    def build(self, input_shape):
        for param_name in self.distributionParam['Params']:
            setattr(self, param_name, self.add_weight(name=param_name, shape=(1,), initializer='random_normal', trainable=True))

    def call(self, inputs):
        mu = getattr(self, 'mu')
        sigma = getattr(self, 'sigma')
        return self.function(inputs, mu, sigma, operator='tf')

# Generalize the Distribution function
class generalizedSingleLayerFunctions (Layer):
    def __init__ (self, distributionParam, function, **kwargs):
        super(generalizedSingleLayerFunctions, self).__init__(**kwargs)
        self.distributionParam = distributionParam
        self.function = function
        pass

    def build(self, input_shape):
        for param_name in self.distributionParam['Params']:
            setattr(self, param_name, self.add_weight(name=param_name, shape=(1,), initializer='random_normal', trainable=True))

    def call(self, inputs):
        params = list()
        for param_name in self.distributionParam['Params']:
            params.append(getattr(self, param_name))
        return self.function(inputs, [param for param in params], operator='tf')


class LayerFunctionMultiple (Layer):
    def __init__ (self, distributionParam, function, **kwargs):
        super(LayerFunctionMultiple, self).__init__(**kwargs)
        self.distributionParam = distributionParam
        self.function = function
        pass

    def build(self, input_shape):
        for param_name in self.distributionParam['Params']:
            setattr(self, param_name, self.add_weight(name=param_name, shape=(1,), initializer='random_normal', trainable=True))

    # Here is required to include an array of parameters and "unpack" them
    def call(self, inputs):
        mus = []
        sigmas = []
        for param in self.distributionParam['Params']:
            if 'mu' in param:
                mus.append(getattr(self, param))
            if 'sigma' in param:
                sigmas.append(getattr(self, param))

        return self.function(inputs, mus, sigmas, operator='tf')




