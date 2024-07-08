import math
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import yfinance as yf
import pandas as pd
from tensorflow.keras.layers import Input, Layer
import tensorflow as tf
from Distributions import Distributions as d

# Class created just to implement Layers for NN-based Optimization

class LayerFunctions (Layer):
    def __init__ (self, distributionParam, **kwargs):
        super(LayerFunctions, self).__init__(**kwargs)
        self.distributionParam = distributionParam
        pass

    def build(self, input_shape):
        for param_name in self.distributionParam['Params']:
            setattr(self, param_name, self.add_weight(name=param_name, shape=(1,), initializer='random_normal', trainable=True))

    def call(self, inputs):
        mu = getattr(self, 'mu')
        sigma = getattr(self, 'sigma')
        return d.Distributions(inputs).normalDistributionCDF(sigma, mu, operator='tf')



