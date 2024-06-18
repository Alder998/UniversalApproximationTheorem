import Distributions.Distributions as dis
import Utils.utils as utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Models.NeuralNetworkModel as model

numberOfDistributions = 3
mus = [-1, 0, 1]
sigmas = [0.5, 1, 1.5]

# Define and Plot a function, for example
function = lambda x: dis.Distributions(x).normalMixtureDistributionPDF(numberOfDistributions,
                                                                       mus, sigmas)
# Get the dataset
ySet = utils.utils().functions().generateFunctionByPoints(function, pointStart=-5, pointEnd=5, steps=5000)[1]
xSet = utils.utils().functions().generateFunctionByPoints(function, pointStart=-5, pointEnd=5, steps=5000)[0]

sample = utils.utils.dataSetPreparation(ySet, xSet).trainTestSplit()

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
plt.plot (pd.DataFrame(ySet).set_index(xSet), color = 'red', label="Original Functions Points")
plt.title (f"Approximation: {epochs} Epochs")
plt.legend()
plt.show()