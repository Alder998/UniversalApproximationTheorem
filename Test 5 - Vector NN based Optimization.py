import Distributions.Distributions as dis
import Utils.utils as utils
import matplotlib.pyplot as plt
import pandas as pd
import Models.NNBasedOptimization as model
from DistributionLayers import LayerFunctions as ly

# Train Set: the Selected Stock's Empirical Distribution
# Test Set: A 2 normal mixture distribution, that we can allow to float according the mean and the Volatility
# to find the best path

numberOfDistributions = 3
multiple = numberOfDistributions != 1

# Normal Mixture function
functionParam = dis.Distributions().normalMixtureDistributionCDF(numberOfDistributions, return_params=True)
function = lambda x, mus, sigmas, operator: dis.Distributions(x).normalMixtureDistributionCDF(numberOfDistributions,
                                                                        mus, sigmas, operator)
layerAssociatedToFunction = ly.LayerFunctionMultiple(functionParam, function)

# Target function that we aim to approximate
functionDataSet = dis.Distributions().empiricalDistributionFromTradedStock('UCG.MI','4d', '1m')
sample = utils.utils.dataSetPreparation(functionDataSet[1], functionDataSet[0]).trainTestSplit()

epochs = 50
modelPrediction = model.functionLayer(layerAssociatedToFunction).buildModel(sample[0], sample[1], sample[2],
                                                                        sample[3], epochs=epochs, multiple=multiple)

# Plot the prediction against the actual function representation
plt.figure(figsize = (12, 5))
plt.scatter (x=sample[2], y=modelPrediction, label="Prediction")
plt.plot (pd.DataFrame(functionDataSet[1]).set_index(functionDataSet[0]), color = 'red', label="Original Functions Points")
plt.title (f"Approximation: {epochs} Epochs")
plt.legend()
plt.show()