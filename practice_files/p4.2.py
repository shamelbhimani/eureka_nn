# Classes and Objects with Neural Network Layers

import numpy as np

X = [
    [1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
]


class LayerDense:
    def __init__(self, n_inputs: int, n_neurons: int) -> None:
        self.output = None
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs: object | list[list[int | float]]) -> None:
        self.output = np.dot(inputs, self.weights) + self.biases


try:
    layer_1 = LayerDense(4, 5)
    layer_2 = LayerDense(5, 2)
    layer_1.forward(X)
    print("Layer 1 output:", layer_1.output)
    layer_2.forward(layer_1.output)
    print("Layer 2 output:", layer_2.output)
except Exception as e:
    print(e)
