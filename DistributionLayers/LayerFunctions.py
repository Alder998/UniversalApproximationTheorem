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

class LayerFunctions:
    def __init__ (self, distributionParam,  **kwargs):
        super(LayerFunctions, self).__init__(**kwargs)
        self.distributionParam = distributionParam
        pass

    def call (self, inputs):
        return d.Distributions(inputs).normalDistributionPDF(self.distributionParam['values'][0], self.distributionParam['values'][1])

    def getLayerFromFunction (self):
        newLayer = Layer()
        for idx in range(len(self.distributionParam['Params'])):
            newLayer.add_weight(name=self.distributionParam['Params'][idx], shape=(1,), initializer='random_normal', trainable=True)
        return newLayer



