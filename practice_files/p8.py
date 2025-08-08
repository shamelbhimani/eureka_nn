import numpy as np
import nnfs
from nnfs.datasets import spiral_data

class LayerDense:
    def __init__(self, n_inputs: int, n_neurons: int) -> None:
        self.inputs = None
        self.d_inputs = None
        self.d_biases = None
        self.d_weights = None
        self.output = None
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs: np.ndarray) -> None:
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues: np.ndarray) -> None:
        self.d_weights = np.dot(self.inputs.T, dvalues)
        self.d_biases = np.sum(dvalues, axis=0, keepdims=True)
        self.d_inputs = np.dot(dvalues, self.weights.T)


class ActivationReLU:
    def __init__(self):
        self.d_inputs = None
        self.inputs = None
        self.output = None

    def forward(self, inputs: np.ndarray) -> None:
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        
    def backward(self, dvalues: np.ndarray) -> None:
        self.d_inputs = dvalues.copy()
        self.d_inputs[self.inputs <= 0] = 0


class ActivationSoftmax:
    def __init__(self):
        self.output = None

    def forward(self, inputs: np.ndarray) -> None:
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Loss:
    def __init__(self):
        self.output = None

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

    def forward(self, output: np.ndarray, y) -> np.ndarray:
        pass

    def backward(self, dvalues: np.ndarray, y) -> None:
        pass

class CategoricalCrossEntropy(Loss):
    def __init__(self):
        super().__init__()
        self.d_inputs = None

    def forward(self, yhat, y_true) -> np.ndarray:
        samples = len(yhat)
        yhat_clipped = np.clip(yhat, a_min= 1e-7, a_max=1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = yhat_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(yhat_clipped * y_true, axis=1)

        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood

    def backward(self, dvalues, y_true) -> None:
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.d_inputs = -y_true/dvalues
        self.d_inputs = self.d_inputs / samples


class Activation_Softmax_Loss_CategoricalCrossentropy:
    def __init__(self):
        self.output = None
        self.d_inputs = None
        self.activation = ActivationSoftmax()
        self.loss = CategoricalCrossEntropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.d_inputs = dvalues.copy()
        self.d_inputs[range(samples), y_true] -= 1
        self.d_inputs = self.d_inputs / samples


class OptimizerSGD:
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                 (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer: LayerDense) -> None:
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.d_weights

            bias_updates = self.momentum * layer.bias_momentums - \
                           self.current_learning_rate * layer.d_biases

            layer.weight_momentums = weight_updates
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.d_weights
            bias_updates = -self.current_learning_rate * layer.d_biases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1

nnfs.init()
X, y = spiral_data(100, 3)

dense1 = LayerDense(2, 64)
act1 = ActivationReLU()
dense2 = LayerDense(64, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
optimizer = OptimizerSGD(decay= 1e-3, momentum=0.5)

for epoch in range(10001):
    dense1.forward(X)
    act1.forward(dense1.output)
    dense2.forward(act1.output)
    loss = loss_activation.forward(dense2.output, y)
    predictions = np.argmax(loss_activation.output, axis=1)
    y_labels = y
    if len(y_labels.shape) == 2:
        y_labels = np.argmax(y_labels, axis=1)
    accuracy = np.mean(predictions == y_labels)

    if not epoch % 100:
        print(f'Epoch: {epoch} '
              f'acc: {accuracy:.3f} '
              f'loss: {loss:.3f} '
              f'LR: {optimizer.current_learning_rate}')

    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.d_inputs)
    act1.backward(dense2.d_inputs)
    dense1.backward(act1.d_inputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()