import numpy as np

from layer.pooling_layer import MaxPooling

input_data = np.array([[[2, 1, 1, 3],
                       [1, 1, 1, 1],
                       [1, 1, 1, 1],
                       [4, 1, 1, 5]]])

pool = MaxPooling((2, 2))

output = pool.forward_propagation(input_data)

print(output)

output_error = np.array([[[2, 3], [4, 5]]])
input_error = pool.backward_propagation(output_error, 1)

print(input_error)