import BasicFunctions.Functions as bf
import Utils.utils as utils
import matplotlib.pyplot as plt
import numpy as np
import BasicFunctions.NeuralNeworkModel as nn

# Define and Plot a sine function, for example
function = lambda x: bf.Functions(x).cosineFunction()
# Get the dataset
ySet = utils.utils().functions().generateFunctionByPoints(function, steps=5000)[1]
xSet = utils.utils().functions().generateFunctionByPoints(function, steps=5000)[0]

sample = utils.utils.dataSetPreparation(ySet, xSet).trainTestSplit()

# Get the dataset
TrainX = np.array(sample[0])
TrainY = np.array(sample[2])
TestX = np.array(sample[1])
TestY = np.array(sample[3])

# Now, Predict the values with a NN Model
epochs = 500
modelPrediction = nn.NNModel().trainAndEvaluateNNModel(TrainX, TrainY, TestX, TestY, epochs = epochs)

# Plot the prediction against the actual function representation
plt.figure(figsize = (12, 5))
plt.scatter (x=TestX, y=modelPrediction, label="Prediction")
plt.scatter (x=xSet, y=ySet, color = 'red', label="Original Functions Points")
plt.title (f"Approximation: {epochs} Epochs")
plt.legend()
plt.show()