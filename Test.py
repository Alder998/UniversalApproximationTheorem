import BasicFunctions.Functions as bf
import Utils.utils as utils

# Define and Plot a sine function
functionTest = bf.Functions()
representation = utils.utils().functions().generateFunctionByPoints(functionTest.sineFunction)

print(representation)