import numpy as np
import nnfs
from numpy import floating
from nnfs.datasets import spiral_data
from abc import ABC, abstractmethod


class LayerDense:
    def __init__(self, n_inputs: int, n_neurons: int) -> None:
        self.inputs = None
        self.outputs = None
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        #Partial Derivatives initialization
        self.d_inputs = None
        self.d_weights = None
        self.d_biases = None

    def forward(self, inputs: np.ndarray) -> None:
        self.inputs = inputs.copy()
        self.outputs = np.dot(self.inputs, self.weights) + self.biases

    def backward(self, dl_dz: np.ndarray) -> None:
        self.d_weights = np.dot(self.inputs.T, dl_dz)
        self.d_biases = np.sum(dl_dz, axis=0, keepdims=True)
        self.d_inputs = np.dot(dl_dz, self.weights.T)


class Activation(ABC):
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.d_inputs = None

    def calculate(self, inputs: np.ndarray):
        self.inputs = inputs.copy()
        self.outputs = self.forward(inputs)

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, dl_dz: np.ndarray) -> np.ndarray:
        pass


class ReLU(Activation):
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        output_array = np.maximum(0, inputs)
        return output_array

    def backward(self, dl_dz: np.ndarray) -> np.ndarray:
        self.d_inputs = dl_dz.copy()
        self.d_inputs[self.inputs <= 0] = 0
        return self.d_inputs

class Softmax(Activation):
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        numerator = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        denominator = np.sum(numerator, axis=1, keepdims=True)
        return numerator / denominator

    def backward(self, dl_dz: np.ndarray) -> np.ndarray:
        self.d_inputs = dl_dz.copy()
        return self.d_inputs


class Loss(ABC):
    def __init__(self):
        self.yhat = None
        self.y = None
        self.d_inputs = None
        self.loss_value = None

    def calculate(self, yhat: np.ndarray,
                  y: np.ndarray | list[int]) -> floating:
        self.yhat = yhat.copy()
        self.y = y.copy()

        sample_losses = self.forward(yhat, y)
        mean_batch_losses = np.mean(sample_losses)
        self.loss_value = mean_batch_losses
        return mean_batch_losses

    @abstractmethod
    def forward(self, yhat: np.ndarray,
                y: np.ndarray | list[int]) -> float:
        pass

    @abstractmethod
    def backward(self, yhat: np.ndarray,
                 y: np.ndarray | list[int]) -> None:
        pass


class CategoricalCrossEntropyLoss(Loss):
    def forward(self, yhat: np.ndarray,
                y: np.ndarray | list[int]) -> float:
        samples = len(yhat)
        yhat_clipped = np.clip(yhat, 1e-7, 1-1e-7)

        if y.ndim == 1:
            correct_confidences = yhat_clipped[range(samples), y]
        elif y.ndim == 2:
            correct_confidences = np.sum(yhat_clipped*y, axis=1)
        else:
            raise ValueError('Array must be of depth 1D or 2D.')

        negative_logs_likelihood = -np.log(correct_confidences)
        return np.mean(negative_logs_likelihood)

    def backward(self, yhat: np.ndarray,
                 y: np.ndarray | list[int]) -> None:
        samples = len(yhat)

        if y.ndim == 1:
            y_encoded = np.zeros_like(y)
            y_encoded[range(samples), y] = 1
        if y.ndim == 2:
            y_encoded = y.copy()
        else:
            raise ValueError('Array must be of depth 1D or 2D.')

        self.d_inputs = (yhat - y_encoded) / samples


class Optimizer(ABC):
    def __init__(self, learning_rate: float=0.05,
                 decay_rate: float=0.001, momentum: float=0.0) -> None:
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self) -> None:
        if self.decay_rate:
            self.current_learning_rate = self.learning_rate * \
                                         (1/(1+self.decay_rate*self.iterations))

    @abstractmethod
    def update_params(self, layer: LayerDense) -> None:
        pass

    def post_update_params(self) -> None:
        self.iterations += 1


class StochasticGradientDescent(Optimizer):
    def update_params(self, layer: LayerDense) -> None:
        if self.momentum != 0:
            if not hasattr(layer, 'weight_momentum'):
                layer.weight_momentum = np.zeros_like(layer.weights)
                layer.biases_momentum = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weight_momentum - \
                self.current_learning_rate * layer.d_weights
            layer.weight_momentum = weight_updates

            bias_updates = self.momentum * layer.biases_momentum - \
                        self.current_learning_rate * layer.d_biases
            layer.biases_momentum = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.d_weights
            bias_updates = -self.current_learning_rate * layer.d_biases

        layer.weights += weight_updates
        layer.biases += bias_updates