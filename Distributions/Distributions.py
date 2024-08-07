import math
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import yfinance as yf
import pandas as pd
from tensorflow.keras.layers import Input, Layer
import tensorflow as tf
import scipy.special
from Utils.utils import utils


# It is not possible, as it is common belief, to approximate a distribution as
# it will be a generic function (i.e. generating random points, and making our net "Following" them). We can try to
# find alternative paths for our Net

class Distributions:
    def __init__ (self, x=0):
        self.x = x
        pass

    # Now the normal PDF must be implemented: it is: 1/(sqrt(2pi * sigma**2)) * e^((-1/2)*((x - mu) / sigma)**2)
    def normalDistributionPDF(self, sigma=1, mu=0, operator='np', return_params=False):
        if (return_params):
            params = {'Params': ['mu', 'sigma'], 'values': [mu, sigma]}
            return params
        if operator == 'np':
            closedForm = 1 / (np.sqrt(2*np.pi * sigma ** 2)) * np.exp((-1 / 2) * ((self.x - mu) / sigma) ** 2)
        if operator == 'tf':
            closedForm = 1 / (tf.sqrt(2 * np.pi * sigma ** 2)) * tf.exp((-1 / 2) * ((self.x - mu) / sigma) ** 2)
        return closedForm

    # Now we are implementing the Normal Distribution CDF, with the formula 1/2 * (1 + erf((x-mu) / sigma*sqrt(2)))
    def normalDistributionCDF(self, sigma=1, mu=0, operator='np', return_params=False):
        if (return_params):
            params = {'Params': ['mu', 'sigma'], 'values': [mu, sigma]}
            return params
        if operator == 'np':
            closedForm = 1/2 * (1 + math.erf((self.x - mu) / sigma * np.sqrt(2)))
        if operator == 'tf':
            self.x = tf.cast(self.x, tf.float32)
            closedForm = 1/2 * (1 + tf.math.erf((self.x - mu) / sigma * tf.sqrt(2.0)))
        return closedForm

    # Let's define a normal Mixture distribution, therefore a distribution that resemble a product of known
    # Distributions
    def normalMixtureDistributionPDF (self, numberOfDistributions, mus, sigmas, return_params=False):
        if (return_params):
            musName = ['mu' + str(value) for value in np.arange (0, numberOfDistributions)]
            sigmasName = ['sigma' + str(value) for value in np.arange (0, numberOfDistributions)]
            params = {'Distribution': 'Normal',
                      'Params': musName + sigmasName,
                      'values': mus + sigmas}
            return params
        closedForm = list()
        for singleDistribution in range(numberOfDistributions):
            closedFormI = (1 / (np.sqrt(2 * np.pi * sigmas[singleDistribution] ** 2)) *
                           np.exp((-1 / 2) * ((self.x - mus[singleDistribution]) / sigmas[singleDistribution]) ** 2))
            closedForm.append(closedFormI)

        # get the product
        finalClosedForm = np.prod(np.array(closedForm))

        return finalClosedForm

    # Mixture CDF
    def normalMixtureDistributionCDF (self, numberOfDistributions, mus = [], sigmas = [], operator = 'np', return_params=False):
        if (return_params):
            musName = {'mus': ['mu' + str(value) for value in np.arange (0, numberOfDistributions)]}
            sigmasName = {'sigmas': ['sigma' + str(value) for value in np.arange(0, numberOfDistributions)]}
            params = {'Distribution': 'Normal',
                      'Params': {
                        'mus': musName['mus'],
                        'sigmas': sigmasName['sigmas']
                        }
                      }
            return params
        closedForm = list()
        for singleDistribution in range(numberOfDistributions):
            if operator == 'np':
                closedFormI = 1 / 2 * (1 + math.erf((self.x - mus[singleDistribution]) / sigmas[singleDistribution] * np.sqrt(2)))
            if operator == 'tf':
                self.x = tf.cast(self.x, tf.float32)
                closedFormI = 1 / 2 * (1 + tf.math.erf((self.x - mus[singleDistribution]) / sigmas[singleDistribution] * tf.sqrt(2.0)))
            closedForm.append(closedFormI)

        # Stack the closedForm list to create a tensor of shape (None, numberOfDistributions)
        stackedClosedForm = tf.stack(closedForm, axis=1)
        # get the product
        finalClosedForm = tf.reduce_prod(stackedClosedForm, axis=1)
        # reshape the layer
        finalClosedForm = tf.reshape(finalClosedForm, (-1, 1))

        return finalClosedForm

    def studentsTDistributionCDF (self, v = [10], operator = 'np', return_params = False):
        if return_params:
            params = {'Distribution': 'StudentsT',
                      'Params': ['v']
                      }
            return params
        if operator == 'np':
            for singleParam in v:
                closedForm = 1/2 + self.x * ( ((scipy.special.gamma( (singleParam+1)/2 )) / (np.sqrt(np.pi * singleParam) * (scipy.special.gamma(singleParam/2))))
                                          * scipy.special.hyp2f1(1/2, (singleParam+1)/2, 3/2, -(self.x / singleParam)) )
        if operator == 'tf':
            for singleParam in v:

                singleParam = tf.cast(singleParam, tf.float32)
                self.x = tf.cast(self.x, tf.float32)

                x2 = tf.square(self.x)
                denom = x2 + singleParam
                reg_beta = utils.functions().regularized_beta(x2 / denom, 0.5 * singleParam, 0.5)

                # Calcola la CDF utilizzando la relazione con la funzione Beta regolarizzata
                closedForm = 0.5 + tf.sign(self.x) * 0.5 * reg_beta
                closedForm = tf.reshape(closedForm, (-1, 1))

        return closedForm

    # Start with Finance (finally): function to create the empirical Distribution function starting from the % returns
    # of a stock

    def empiricalDistributionFromTradedStock (self, ticker, period, interval, datasetStart=-1, datasetEnd=1,
                                              datasetSteps=5000):

        print('Getting Stock Data...')
        returnSample = (yf.Ticker(ticker).history(period, interval)['Close'].pct_change() * 100).dropna()
        print('\n')
        print('Stock Data gotten from source! Time Series size:', len(returnSample))
        #returnSample = pd.read_excel(r"C:\Users\alder\Desktop\AAPL.xlsx")
        ecdf = sm.distributions.ECDF(np.array(returnSample))

        flatSample = np.linspace(datasetStart, datasetEnd, datasetSteps)
        fittedECDF = ecdf(flatSample)

        return [flatSample, fittedECDF]

    # This function is to create mixture distributions from any possible distribution
    def distributionWrapper (self, distributionFunctions, params_list=[], return_params = False):
        combined_params = {i + 1: d for i, d in enumerate(distributionFunctions)}
        if return_params:
            return combined_params

        convertedParams = utils.functions().convert_params_to_float(params_list)

        # Total params
        tot_params = list()
        for value in params_list:
            tot_params.append(params_list[value]['Params'])

        results = []
        for func, params in zip(distributionFunctions, tot_params):
            results.append(func(self.x, params[0], 'tf'))
            print('Distribution appended')

        # Stack the results list to create a tensor
        stackedClosedForm = tf.stack(results, axis=0)
        # get the product
        finalClosedForm = tf.reduce_prod(stackedClosedForm, axis=0)
        # reshape the layer
        finalClosedForm = tf.reshape(finalClosedForm, (-1, 1))

        return finalClosedForm
