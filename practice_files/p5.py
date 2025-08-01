import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class LayerDense:
    def __init__(self, n_inputs: int, n_neurons: int) -> None:
        self.output = None
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward (self, inputs: np.ndarray) -> None:
        self.output = np.dot(inputs, self.weights) + self.biases


class ActivationReLU:
    def __init__(self) -> None:
        self.output = None


    def forward (self, inputs: np.ndarray | list[list[int | float]]) -> None:
        self.output = np.maximum(0, inputs)