# Using the nnfs dataset, we will create an activation function to pass our
# neural network data through.

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()
X, y = spiral_data(100, 3)


class LayerDense:
    def __init__(self, n_inputs: int, n_neurons: int) -> None:
        self.output = None
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs: object | list[list[int | float]]) -> None:
        self.output = np.dot(inputs, self.weights) + self.biases

class ActivationReLU:
    def __init__(self):
        self.output = None

    def forward(self, inputs: object | list[list[int | float]]) -> None:
        self.output = np.maximum(0, inputs)

try:
    layer1 = LayerDense(2, 5)
    activation1 = ActivationReLU()
    layer1.forward(X)
    activation1.forward(layer1.output)

    layer2 = LayerDense(5, 5)
    activation2 = ActivationReLU()
    layer2.forward(activation1.output)
    activation2.forward(layer2.output)
    print(activation2.output)
except Exception as e:
    print(e)
