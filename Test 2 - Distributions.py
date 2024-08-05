import Distributions.Distributions as dis
import Utils.utils as utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Models.NeuralNetworkModel as model

mus = [0, 0.03]
sigmas = [0.7, 1.5]
numberOfDistributions = len(mus)

# Define and Plot a function, for example
function = lambda x: dis.Distributions(x).normalMixtureDistributionCDF(numberOfDistributions,
                                                                       mus, sigmas)
# Get the dataset
dataset = utils.utils().functions().generateFunctionByPoints(function, pointStart=-5, pointEnd=5, steps=5000)

sample = utils.utils.dataSetPreparation(dataset[1], dataset[0]).trainTestSplit()

# Try the simplest approach possible. It is not possible, as it is common belief, to approximate a distribution as
# it will be a generic function (i.e. generating random points, and making our net "Following" them). We can try to
# find alternative paths for our Net

# Now, Predict the values with a NN Model
epochs = 50
modelPrediction = model.NNModel().trainAndEvaluateNNModelForBasicFunctions(sample[0], sample[1], sample[2],
                                                                        sample[3], epochs = epochs)

# Plot the prediction against the actual function representation
plt.figure(figsize = (12, 5))
plt.scatter (x=sample[2], y=modelPrediction, label="Prediction")
plt.plot (pd.DataFrame(dataset[1]).set_index(dataset[0]), color = 'red', label="Original Functions Points")
plt.title (f"Approximation: {epochs} Epochs")
plt.legend()
plt.show()