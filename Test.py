import BasicFunctions.Functions as bf
import Utils.utils as utils

# Define and Plot a sine function
functionTest = bf.functions().sineFunction()
representation = utils.utils().functions(functionTest).generateFunctionByPoints()

print(representation)