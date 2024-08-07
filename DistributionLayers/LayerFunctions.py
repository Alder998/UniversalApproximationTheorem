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
        for key, wrapper in self.distributionParam['Params'].items():
            for param_name in wrapper:
                setattr(self, param_name, self.add_weight(name=param_name, shape=(1,), initializer='random_normal', trainable=True))

    # Here is required to include an array of parameters and "unpack" them
    def call(self, inputs):
        mus = []
        sigmas = []
        print(self.distributionParam['Params'])
        for key, wrapper in self.distributionParam['Params'].items():
            if self.distributionParam['Distribution'] == 'Normal':
                if key == 'mus':
                    for param in wrapper:
                        mus.append(getattr(self, param))
                if key == 'sigmas':
                    for param in wrapper:
                        sigmas.append(getattr(self, param))

        return self.function(inputs, mus, sigmas, operator='tf')

class generalizedLayerFunctionMultiple (Layer):
    def __init__ (self, distributionParam, function, **kwargs):
        super(generalizedLayerFunctionMultiple, self).__init__(**kwargs)
        self.distributionParam = distributionParam
        self.function = function
        pass

    def build(self, input_shape):
        for key, wrapper in self.distributionParam['Params'].items():
            for param_name in wrapper:
                setattr(self, param_name, self.add_weight(name=param_name, shape=(1,), initializer='random_normal', trainable=True))

    # Here is required to include an array of parameters and "unpack" them
    def call(self, inputs):
        mus = []
        sigmas = []
        vs = []
        for param in self.distributionParam['Params']:
            if self.distributionParam['Distribution'] == 'Normal':
                for mu in param['mus']:
                    mus.append(getattr(self, mu))
                for sigma in param['sigmas']:
                    sigmas.append(getattr(self, sigma))
                functionNormal = self.function(inputs, mus, sigmas, operator='tf')
            if self.distributionParam['Distribution'] == 'StudentsT':
                for mu in param['mus']:
                    vs.append(getattr(self, mu))
                functionT = self.function(inputs, vs, operator='tf')

        return self.function(inputs, [functionNormal, functionT], operator='tf')


