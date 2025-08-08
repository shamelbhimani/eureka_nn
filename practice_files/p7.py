import numpy as np
from abc import ABC, abstractmethod
from numpy import floating

class LayerDense:
    def __init__(self, n_inputs: int, n_neurons: int) -> None:
        self.output = None
        self.inputs = None
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # Derivatives
        self.d_inputs = None
        self.d_weights = None
        self.d_biases = None

    def forward(self, inputs: np.ndarray) -> None:
        self.inputs = inputs.copy()
        self.output = np.dot(self.inputs, self.weights) + self.biases

    def backward(self, dl_dz: np.ndarray) -> None:
        self.d_inputs = np.dot(dl_dz, self.weights.T)
        self.d_weights = np.dot(self.inputs.T, dl_dz)
        self.d_biases = np.sum(dl_dz, axis=0, keepdims=True)


class Activation(ABC):
    def __init__(self) -> None:
        self.inputs = None
        self.output = None
        # Derivatives
        self.d_inputs = None

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> None:
        pass

    @abstractmethod
    def backward(self, dl_dz: np.ndarray) -> None:
        pass

    def calculate(self, inputs: np.ndarray) -> None:
        self.inputs = inputs.copy()
        self.output = self.forward(inputs)


class ReLU(Activation):
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        output = np.maximum(0, inputs)
        return output

    def backward(self, dl_dz: np.ndarray) -> None:
        self.d_inputs = dl_dz.copy()
        self.d_inputs[self.inputs <= 0] = 0


class Softmax(Activation):
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        numerator = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        denominator = np.sum(numerator, axis=1, keepdims=True)
        return numerator / denominator

    def backward(self, dl_dz: np.ndarray) -> None:
        pass


class Loss(ABC):
    def __init__(self) -> None:
        self.yhat = None
        self.y = None
        self.d_inputs = None

    def calculate(self,
                  yhat: np.ndarray,
                  y: np.ndarray) -> floating:
        self.yhat = yhat.copy()
        self.y = y.copy()

        sample_losses = self.forward(yhat, y)
        mean_batch_losses = np.mean(sample_losses)
        return mean_batch_losses

    @abstractmethod
    def forward(self, yhat: np.ndarray, y: np.ndarray) -> float:
        pass

    @abstractmethod
    def backward(self, yhat: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass


class CategoricalCrossEntropyLoss(Loss):
    def forward(self, yhat: np.ndarray, y: np.ndarray) -> float:
        samples = len(yhat)
        yhat_clipped = np.clip(yhat, a_min=1e-7, a_max=1-1e-7)

        if len(y.shape) == 1:
            correct_confidences = yhat_clipped[range(samples), y]
        elif len(y.shape) == 2:
            correct_confidences = np.sum(yhat_clipped * y, axis=1)
        else:
            raise ValueError("y must be either a 1d or 2d array")

        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood

    def backward(self, yhat: np.ndarray, y: np.ndarray) -> None:
        samples = len(yhat)

        if len(y.shape) == 1:
            y_encoded = np.zeros_like(yhat)
            y_encoded[range(samples), y] = 1
        elif len(y.shape) == 2:
            y_encoded = y
        else:
            raise ValueError("y must be either a 1d or 2d array")

        self.d_inputs = (yhat - y_encoded) / samples


class Optimizer(ABC):
    def __init__(self, learning_rate: float=0.05,
                 decay_rate: float=0.001) -> None:
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay_rate
        self.iterations = 0

    def pre_update_params(self) -> None:
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                         (1.0/(1+self.decay*self.iterations))

    @abstractmethod
    def update_params(self, layer: LayerDense) -> None:
        pass

    def post_update_params(self) -> None:
        self.iterations += 1


class GradientDescent(Optimizer):
    def update_params(self, layer: LayerDense) -> None:
        layer.weights -= self.learning_rate * layer.d_weights
        layer.biases -= self.learning_rate * layer.d_biases

