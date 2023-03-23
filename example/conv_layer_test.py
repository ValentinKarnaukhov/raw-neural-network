import numpy as np
from layer.convolutional_layer import ConvolutionalLayer

layer1 = ConvolutionalLayer((1, 2, 2), (3, 3, 2))
layer1.weights = np.array([[[[1, 10], [2, 20]], [[4, 40], [5, 50]]]])

layer2 = ConvolutionalLayer((1, 3, 3), (3, 3, 2))
layer2.weights = np.array([[[[1, 10], [2, 20], [3, 30]], [[4, 40], [5, 50], [6, 60]], [[7, 70], [8, 80], [9, 90]]]])

input_data = np.array([[[1, 11], [2, 22], [3, 33]], [[4, 44], [5, 55], [6, 66]], [[7, 77], [8, 88], [9, 99]]])

print(layer1.forward_propagation(input_data))
print(layer2.forward_propagation(input_data))

# layer3 = ConvolutionalLayer((1, 2, 2), (5, 5, 2))
# layer3.weights = np.array([[[[1], [2]], [[3], [4]]]])

# layer3 = ConvolutionalLayer((1, 3, 3), (5, 5, 2))
# layer3.weights = np.array([[[[1, 10], [2, 20], [3, 30]], [[4, 40], [5, 50], [6, 60]], [[7, 70], [8, 80], [9, 90]]]])


# input_data = np.array([[[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
#                        [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
#                        [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
#                        [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]],
#                        [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]])
#
# output_data = layer3.forward_propagation(input_data)
# print(output_data)
#
# input_gradient = layer3.backward_propagation(output_data, 1)
# print(input_gradient)
