import Distributions.Distributions as dis
import Utils.utils as utils
import matplotlib.pyplot as plt
import pandas as pd
import Models.NNBasedOptimization as model
from DistributionLayers import LayerFunctions as ly

# Train Set: the Selected Stock's Empirical Distribution
# Test Set: Try to stak two distributions: one Normal and one Student's T

# Generalized Mixture Distribution - Params
functionParamWrapper = dis.Distributions().distributionWrapper(distributionFunctions = [
    dis.Distributions().studentsTDistributionCDF(return_params=True),
    dis.Distributions().normalDistributionCDF(return_params=True)
], return_params = True)

# General useful params
numberOfDistributions = len(functionParamWrapper)
multiple = numberOfDistributions != 1

# Generalized Mixture Distribution - Functions
functionWrapper = dis.Distributions().distributionWrapper(distributionFunctions = [
    lambda x, v, operator: dis.Distributions(x).studentsTDistributionCDF(v, operator),
    lambda x, mu, sigma, operator: dis.Distributions(x).normalDistributionCDF(mu, sigma, operator),
], params_list=functionParamWrapper)

layerAssociatedToFunction = ly.generalizedLayerFunctionMultiple(functionParamWrapper, functionWrapper)

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