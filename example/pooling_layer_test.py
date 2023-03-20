import numpy as np

from layer.pooling_layer import MaxPooling

input_data = np.array([[2, 1, 1, 3],
                       [1, 1, 1, 1],
                       [1, 1, 1, 1],
                       [4, 1, 1, 5]])

pool = MaxPooling((2, 2))

output = pool.forward_propagation(input_data)

print(output)