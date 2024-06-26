import BasicFunctions.Functions as bf
import Utils.utils as utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Models.NeuralNetworkModel as model

# Define and Plot a sine function, for example
function = lambda x: bf.Functions(x).cosineFunction()
# Get the dataset
ySet = utils.utils().functions().generateFunctionByPoints(function, steps=5000)[1]
xSet = utils.utils().functions().generateFunctionByPoints(function, steps=5000)[0]

sample = utils.utils.dataSetPreparation(ySet, xSet).trainTestSplit()

# Now, Predict the values with a NN Model
epochs = 700
modelPrediction = model.NNModel().trainAndEvaluateNNModelForBasicFunctions(sample[0], sample[1], sample[2],
                                                                        sample[3], epochs = epochs)

# Plot the prediction against the actual function representation
plt.figure(figsize = (12, 5))
plt.scatter (x=sample[2], y=modelPrediction, label="Prediction")
plt.plot (pd.DataFrame(ySet).set_index(xSet), color = 'red', label="Original Functions Points")
plt.title (f"Approximation: {epochs} Epochs")
plt.legend()
plt.show()