import BasicFunctions.Functions as bf
import Utils.utils as utils
from functools import partial

# Define and Plot a sine function, for example
function = lambda x: bf.Functions(x).sineFunction()
# Get the dataset
ySet = utils.utils().functions().generateFunctionByPoints(function)[1]
xSet = utils.utils().functions().generateFunctionByPoints(function)[0]

sample = utils.utils.dataSetPreparation(ySet, xSet).trainTestSplit()

# Get the dataset
TrainX = sample[0]
TrainY = sample[2]
TestX = sample[1]
TestY = sample[3]

# Now, we add the Model Test
print(TrainX)