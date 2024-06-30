import Distributions.Distributions as dis
import Utils.utils as utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Models.NeuralNetworkModel as model

# Train Set: the Selected Stock's Empirical Distribution
# Test Set: A 2 normal mixture distribution, that we can allow to float according the mean and the Volatility
# to find the best path

numberOfDistributions = 1

# Define and Plot a function, for example
function = lambda x, mu, sigma: dis.Distributions(x).normalMixtureDistributionPDF(numberOfDistributions, mu, sigma)