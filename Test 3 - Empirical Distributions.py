import Distributions.Distributions as dis
import Utils.utils as utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Models.NeuralNetworkModel as model


# Define and Plot a function, for example
functionDataSet = dis.Distributions().empiricalDistributionFromTradedStock('UCG.MI','1d', '1m')

sample = utils.utils.dataSetPreparation(functionDataSet[1], functionDataSet[0]).trainTestSplit()

# Try the simplest approach possible. It is not possible, as it is common belief, to approximate a distribution as
# it will be a generic function (i.e. generating random points, and making our net "Following" them). We can try to
# find alternative paths for our Net

# Now, Predict the values with a NN Model
epochs = 100
modelPrediction = model.NNModel().trainAndEvaluateNNModelForBasicFunctions(sample[0], sample[1], sample[2],
                                                                        sample[3], epochs = epochs)

# Plot the prediction against the actual function representation
plt.figure(figsize = (12, 5))
plt.scatter (x=sample[2], y=modelPrediction, label="Prediction")
plt.plot (pd.DataFrame(functionDataSet[1]).set_index(functionDataSet[0]), color = 'red', label="Original Functions Points")
plt.title (f"Approximation: {epochs} Epochs")
plt.legend()
plt.show()