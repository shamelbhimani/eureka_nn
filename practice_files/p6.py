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
# To simplify our tasks:
#   1. We have to calculate gradient of the loss with respect to the
#   weights and biases in said layer;
#   2. We have to calculate the gradient of the loss with respect to the
#   input of that layer.
# We will do this within the backward function of the LayerDense class.
#
# The gradient of the loss with respect to weights can be calculated using
# this simple formula:
#                   X^T . dL/dZ
#   Where:
#       X^T is the transposition of the inputs matrix
#       dl/dz is the matrix of the gradient of loss with respect to all
#       neurons in that layer.
#
# The gradient of the loss with respect to biases can be calculated using
# this simple formula:
#
#                   np.sum(dL/dZ, axis=0, keepdims=True)
# Which is the sum of all rows of dL/dZ. keepdims=True is necessary to
# prevent the numpy summation method from flattening our 2D array to a 1d
# array or scalar.
#
# The gradients of the loss with respect to inputs can be calculated using
# this simple formula:
#
#                   dl/dZ . W^T
#   Where:
#       dl/dz is the matrix of the gradient of loss with respect to all
#       neurons in that layer.
#       W^T is the transposition of the matrix of the weights of all neurons
#       in that layer.




class LayerDense:
    def __init__(self, n_inputs: int, n_neurons: int) -> None:
        self.inputs = None
        self.output = None
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        # Partial Derivatives storage:
        self.d_weights = None
        self.d_biases = None
        self.d_inputs = None

    def forward(self, inputs: np.ndarray | list[list[int | float]]) -> None:
        """
        The dot product between the initial inputs X of this layer, or the
        outputs of a previous layer Z_i (if not activated), or the output of an
        activation function A_i, is added to the biases associated with this
        layer.
        """
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weights) + self.biases

    def backward(self, dl_dz: np.ndarray) -> None:
        """
        The derivative dL of the output layer with respect to the weights dW
        associated with the transposition of the initial inputs X of this
        layer, or the transposition of the outputs of a previous layer Z_i (
        if not activated), or the transposition of the output of an activation
        A_i is multiplied by the derivative of the loss function dL with
        respect to that loss functions' inputs dZ_{i=last} such that dL/dW =
        A^T_i|Z^T_i|X^T_i . dL/dZ_{i=last} = A^T_i|Z^T_i|X^T_i . (Yhat - Y).

        The derivative of the loss function dL with respect to the biases B_i
        associated with this layer Z_i is the summation over all the errors
        of the output of the network if this layer is Z_{i=last}, i.e.,
        dL/dB_i = Summation dL/dZ_{i=last}^l where l is the number of
        samples. Since the error was calculated to be the derivative of the
        cross-entropy function, and stored as d_inputs of that function,
        we will call that variable as the input to this function.
        However, if this layer is not the output layer, i.e., it is not
        Z_{i=last}, then the derivative of the loss function dL with respect
        to the biases B_i associated with this layer Z_i is the summation
        over the derivative of the loss function dL with respect to the
        outputs of this layer Z_i such that dL/dB_i = Summation dL/dZ_i =
        (dL/dZ_{i+1} . W^T_{i+1}) multiplied by the derivative of the loss
        function dL with respect to the derivative of the activation function
        A_i of this layer. Now, granted that we are only using ReLU
        functions, the derivative of the activation function will either be a 1
        or a 0, so it can simply be ignored. Hence, the final derivative is
        simply the summation of the aforementioned derivative. Since this is
        dependent on the derivatives of the next layer, whether that is a
        hidden or an output layer, we just use the derivative of the next
        layer, i.e., dL/dZ_{i+1}, as the input to this function and sum it to
        form the derivative of biases for this layer.
        """
        self.d_weights = np.dot(self.inputs.T, dl_dz)
        self.d_biases = np.sum(dl_dz, axis=0, keepdims=True)
        self.d_inputs = np.dot(dl_dz, self.weights.T)


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

    @abstractmethod
    def backward(self, d_inputs: np.ndarray) -> None:
        pass


class ReLU(Activation):
    def forward(self, inputs: np.ndarray | list[list[int | float]]) -> None:
        output_array = np.maximum(0, inputs)
        return output_array

    def backward(self, dl_dz: np.ndarray) -> None:
        """
        The backward pass for ReLU at Z_i. It calculates the gradient of the
        loss with respect to the inputs of the ReLU layer (`self.d_inputs`).
        The inputs this function receives will be the derivatives of the loss dL
        with respect to derivatives of layer Z_{i+1} (in the forward sequence)
        dZ_{i+1} such that dL/A_i @ Z_i = dL/dZ{i+1} if > 0, else 0.

        The derivative of the ReLU function is:
        - 1 if the input `x > 0`
        - 0 if the input `x <= 0`

        This means we pass the upstream gradient (`d_inputs`) through unchanged
        for all positive inputs, and we "kill" the gradient (set it to 0)
        for all non-positive inputs.

        We use a copy of the upstream gradient and then apply the mask based on
        the stored inputs from the forward pass.
        """
        self.d_inputs = dl_dz.copy()
        self.d_inputs[self.inputs <= 0] = 0


class Softmax(Activation):
    def forward(self, inputs: np.ndarray) -> None:
        numerator = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        denominator = np.sum(numerator, axis=1, keepdims=True)
        return numerator / denominator

    def backward(self, d_inputs: np.ndarray) -> None:
        # Not implemented since the CategoricalCrossEntropy.backward()
        # calculates gradient of loss with respect to softmax output.
        pass


class Loss(ABC):
    def __init__(self) -> None:
        self.y_pred = None
        self.y_true = None
        self.d_inputs = None

    def calculate(self, y_pred: np.ndarray,
                  y_true: np.ndarray) -> floating[Any]:
        self.y_pred = y_pred
        self.y_true = y_true
        samples_losses = self.forward(y_pred, y_true)
        mean_batch_loss = np.mean(samples_losses)
        return mean_batch_loss

    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        pass

    @abstractmethod
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
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

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> None:
        """
        The derivative of the loss function dL with respect to the inputs, i.e.,
        the output of the output layer dL/dZ_{i = last} = Yhat - Y, making
        d_inputs = d_outputs of the previous layer = Yhat - Y.
        """
        samples = len(y_pred)

        if len(y_true.shape) == 1:
            y_true_one_hot = np.zeros_like(y_pred)
            y_true_one_hot[range(samples), y_true] = 1
        elif len(y_true.shape) == 2:
            y_true_one_hot = y_true
        else:
            raise ValueError("y_true must be a 1D or 2D array.")

        self.d_inputs = (y_pred - y_true_one_hot) / samples


try:
    #Create data
    nnfs.init()
    X, y = spiral_data(100, 3)

    #Create layers and functions
    layer1 = LayerDense(X.shape[1], 32)
    activation_function1 = ReLU()
    layer2 = LayerDense(32, 32)
    activation_function2 = ReLU()
    layer3 = LayerDense(32, 3)
    activation_function3 = Softmax()
    loss_function = CategoricalCrossEntropy()

    #Training Loop starts here:
    for epoch in range(10000000):
    #Forward passing
        layer1.forward(X)
        activation_function1.calculate(layer1.output)

        layer2.forward(activation_function1.output)
        activation_function2.calculate(layer2.output)

        layer3.forward(activation_function2.output)
        activation_function3.calculate(layer3.output)

        # Calculating loss
        loss = loss_function.calculate(activation_function3.output, y)

        # Backpropagation
        loss_function.backward(activation_function3.output, y)
        layer3.backward(loss_function.d_inputs)
        activation_function2.backward(layer3.d_inputs)
        layer2.backward(activation_function2.d_inputs)
        activation_function1.backward(layer2.d_inputs)
        layer1.backward(activation_function1.d_inputs)

        # Weights and Biases updating based on arbitrary learning rate:
        learning_rate = 0.005

        layer3.weights -= learning_rate * layer3.d_weights
        layer3.biases -= learning_rate * layer3.d_biases

        layer2.weights -= learning_rate * layer2.d_weights
        layer2.biases -= learning_rate * layer2.d_biases

        layer1.weights -= learning_rate * layer1.d_weights
        layer1.biases -= learning_rate * layer1.d_biases

        if epoch % 100 == 0:
            print(f'Epoch {epoch} - Loss: {loss:.4f}')
except Exception as e:
    print(e)
