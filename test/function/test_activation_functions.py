import unittest
import numpy as np

from rawnn.function.activation_functions import Sigmoid, Tanh, ReLU
from test import one_dimensional_array


class TestSigmoidFunction(unittest.TestCase):
    def test_sigmoid_function(self):
        actual_result = np.round(Sigmoid.function(one_dimensional_array), decimals=4)
        target_result = np.round(np.array([0.1192, 0.2689, 0.5, 0.7311, 0.8808]), decimals=4)
        np.testing.assert_array_equal(target_result, actual_result)

    def test_sigmoid_function_derivative(self):
        actual_result = np.round(Sigmoid.function_derivative(one_dimensional_array), decimals=4)
        target_result = np.round(np.array([0.1050, 0.1966, 0.25, 0.1966, 0.10499]), decimals=4)
        np.testing.assert_array_equal(target_result, actual_result)


class TestTanhFunction(unittest.TestCase):
    def test_tanh_function(self):
        actual_result = np.round(Tanh.function(one_dimensional_array), decimals=4)
        target_result = np.round(np.array([-0.9640, -0.7616, 0, 0.7616, 0.9640]), decimals=4)
        np.testing.assert_array_equal(target_result, actual_result)

    def test_tanh_function_derivative(self):
        actual_result = np.round(Tanh.function_derivative(one_dimensional_array), decimals=4)
        target_result = np.round(np.array([0.0707, 0.4200, 1, 0.4200, 0.0707]), decimals=4)
        np.testing.assert_array_equal(target_result, actual_result)


class TestReLUFunction(unittest.TestCase):
    def test_relu_function(self):
        actual_result = np.round(ReLU.function(one_dimensional_array), decimals=4)
        target_result = np.round(np.array([0, 0, 0, 1, 2]), decimals=4)
        np.testing.assert_array_equal(target_result, actual_result)

    def test_relu_function_derivative(self):
        actual_result = np.round(ReLU.function_derivative(one_dimensional_array), decimals=4)
        target_result = np.round(np.array([0, 0, 0, 1, 1]), decimals=4)
        np.testing.assert_array_equal(target_result, actual_result)


if __name__ == '__main__':
    unittest.main()
