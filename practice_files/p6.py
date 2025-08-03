import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from abc import ABC, abstractmethod
from numpy import floating
from typing import Any


class LayerDense:
    def __init__(self, n_inputs: int, n_neurons: int) -> None:
        self.output = None
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs: np.ndarray | list[list[int | float]]) -> None:
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation(ABC):
    def __init__(self) -> None:
        self.output = None

    def calculate(self, inputs: np.ndarray | list[list[int | float]]) -> None:
        self.output = self.forward(inputs)

    @abstractmethod
    def forward(self, inputs: np.ndarray | list[list[int | float]]) -> None:
        pass


class ReLU(Activation):
    def forward(self, inputs: np.ndarray | list[list[int | float]]) -> None:
        output_array = np.maximum(0, inputs)
        return output_array


class Softmax(Activation):
    def forward(self, inputs: np.ndarray) -> None:
        numerator = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        denominator = np.sum(numerator, axis=1, keepdims=True)
        output_array = numerator / denominator
        return output_array


class Loss(ABC):
    def __init__(self) -> None:
        self.y_pred = None
        self.y_true = None

    def calculate(self, y_pred: np.ndarray,
                  y_true: np.ndarray) -> floating[Any]:
        samples_losses = self.forward(y_pred, y_true)
        mean_batch_loss = np.mean(samples_losses)
        return mean_batch_loss

    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        pass


class CategoricalCrossEntropy(Loss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        else:
            raise ValueError("y_true must be a 1D or 2D array.")

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


try:
    nnfs.init()
    X, y = spiral_data(100, 3)
    layer1 = LayerDense(X.shape[1], 5)
    activation_function1 = ReLU()
    layer2 = LayerDense(5, 3)
    activation_function2 = Softmax()
    loss_function = CategoricalCrossEntropy()

    layer1.forward(X)
    activation_function1.calculate(layer1.output)
    layer2.forward(activation_function1.output)
    activation_function2.calculate(layer2.output)

    loss = loss_function.calculate(activation_function2.output, y)
    print(loss)
except Exception as e:
    print(e)
