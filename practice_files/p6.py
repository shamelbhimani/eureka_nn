import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from abc import ABC, abstractmethod
from numpy import floating
from typing import Any

# Backpropagation is an optimization issue.
# 1. Let us minimize the loss function
#   1.1. We need to find how to update the weights (w[0], w[1], w[2]) and
#   biases (b) so that loss is minimized.
#   1.2. For this, we have to move in the negative gradient direction. We have
#   to find the derivative of loss with respect to weights and biases (w[0],
#   w[1], w[2], b). This would look like:
#               dL/dw_0, dL/dw_1, dL/dw_2, dL/db
#
# These would be subtracted from the standing weights and biases such that:
#               w = w - n * dL/dw
#   where n is a step-size
# This would also be the same for the biases;
#               b = b - n * dL/db
#
# Taking a look at the schematic of our operation:
# 1. The dot products of inputs and weights at index i such that i = 1, 2,...,
# n is summed with a neuron's bias such that x_0w_0 + x_1w_1 + ... + x_nw_n +
# bias.
# 2. This summation is then passed through the ReLU activation function at
# for each neuron within the hidden layer that selects the maximum between
# the input value post-summation and 0.
# 3. It is then passed to the output layer neuron that performs the same
# dot product and summation process.
# 4. The output is passed through the loss function to determine the loss.
#
# As we can see, the initial operation has an outcome on the value of the
# loss function. However, it goes through its own operations before reaching
# the loss function, and therefore creates an indirect dependency. It would
# look something like this:
#   Hidden neuron weighted sum:
#       z_h = w_h^T . x + b_h
#
#   ReLU Activation:
#       a_h = ReLU(z_h) = max(0, z_h)
#       a_h = max(0, w_h^T . x + b)                                 substitution
#
#   Output Neuron Weighted Sum:
#       z_{o,k} = a_h . w_{o,k} + b_{o,k}
#       z_{o,k} = (max(0, w_h^T . z + b_h)) . w_{o,k} + b_{o,k}     substitution
#
#   Softmax Activation:
#       yhat_k = e^{Z_{o,k}}/Sum(e^{Z_{o,k}}
#       yhat_k = e^{((max(0,w_h^T . x+b_h)) . w_{o,k} + b_{o,k})
#                ///////////////////////////////////////////////
#                Sum(e^{((max(0,w_h^T.x+b_h)).w_{o,j}+b_{o,j}).     substitution
#
#   Categorical Cross Entropy:
#       L = - Sum(y_k . log(yhat_k))
#       L(x,y) = -Sum(y_k . log(Softmax(Output_weighted_sum(ReLU(
#       hidden_weighted_sum)))))                        final composite function
#
#   Total Dependency Chain:
#       L -> yhat -> z_o -> a_h -> z_h -> {w_h,b_h,w_o,b_o}
#
# Applying Chain Rule:
#           NOTE: When referring to z_o, I mean output neuron output vector.
#           When
#           referring
#           to z_h, I mean hidden neuron output vector
#
#   Before jumping into the detailed breakdown of each function, I will list
#   out simply how the partial derivatives interact to find the gradient of
#   the loss.
#       dL/d{w_h,b_h} = dL/dyhat * dyhat/dz_o * dz_o/da_h * da_h/dz_h *
#       dz_h/d{w_h,b_h}
#
#   The Initial Error (L -> z_o), the derivative of loss L with respect
#   to the output logits vector z_o:
#       dL/dz_0 = yhat - y
#
#   Chaining L -> z_o -> w_o (Gradient with respect to output layer weights w_o)
#       dL/dw_o = dL/dz_o . dz_o/dw_o
#   From the output layer weighted sum, z_o = a_h . w_o + b_o, the partial
#   derivative of z_o with respect to the w_o is the input from the hidden
#   layer, a_h.

#   Substituting our error and this derivative, we get the gradient for each
#   weight w_{o,k}:
#       dL/dw_{o,k} = (yhat_k - y_k) . a_h
#   This tells us the gradient is the error for that output neuron scaled by
#   the output of the hidden neuron.
#
#   Chaining L -> z_o -> b_o (Gradient with respect to output layer biases)
#       dL/db_o = dL/dz_o . dz_o/db_o
#   The partial derivative of z_o with respect to b_o is 1.
#   Substituting, the gradient for each bias b_{o,k}, is simply the error for
#   that output neuron:
#       dL/db_{o,k} = yhat_k - y_k
#
#




class LayerDense:
    def __init__(self, n_inputs: int, n_neurons: int) -> None:
        self.output = None
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.inputs = None
        self.dweights = None
        self.dbiases = None
        self.dinputs = None

    def forward(self, inputs: np.ndarray | list[list[int | float]]) -> None:
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dinputs: np.ndarray) -> None:
        self.dweights = np.dot(self.inputs.T, dinputs)
        self.dbiases = np.sum(dinputs, axis=0, keepdims=True)
        self.dinputs = np.dot(dinputs, self.weights.T)


class Activation(ABC):
    def __init__(self) -> None:
        self.output = None
        self.inputs = None
        self.d_inputs = None

    def calculate(self, inputs: np.ndarray | list[list[int | float]]) -> None:
        self.inputs = inputs
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
