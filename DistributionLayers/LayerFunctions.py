from tensorflow.keras.layers import Input, Layer

# Class created just to implement Layers for NN-based Optimization

class LayerFunctions (Layer):
    def __init__ (self, distributionParam, function, **kwargs):
        super(LayerFunctions, self).__init__(**kwargs)
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

        return self.function(inputs, [mu for mu in mus], [sigma for sigma in sigmas], operator='tf')



