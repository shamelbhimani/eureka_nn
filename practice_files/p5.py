import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Categorical cross-entropy is a fundamental loss function used in neural
# networks, particularly for multi-class classification problems. It's the
# mechanism by  which the network understands how 'wrong' its predictions
# are, allowing it to learn and improve. Imagine a multi-class classification
# task, like identifying an image as a 'dog', 'cat', or 'bird'. The neural
# network's final layer, using softmax activation, outputs a probability
# distribution. For example, for a given image, the output might be [0.1,
# 0.8, 0.1]. This means the model thinks there's a 10% probability it's a
# dog, an 80% probability it's a cat, and a 10% probability it's a bird. To
# evaluate this prediction, we need to compare it to the 'true' answer. In
# machine learning, this true label is often represented using a one-hot
# encoded vector. For our example, if the image is actually a cat, the true
# label would be [0, 1, 0] where the index of the true label would be 'hot'
# or 1.
#
# Categorical cross-entropy measures the difference between these two
# probability distributions: the model's prediction and the true label. The
# formula for a single sample is:
#
#                   L = - summation_{i=1}^{C} y_i . log(yhat_i)
# Where:
#       L is the loss value
#       C is the number of classes (3 in our example)
#       y_i is the true label for class i (either 0 or 1 from one-hot encoding)
#       yhat_i is the predicted probability for class i.
#
# Because the true label vector is one-hot encoded, this formula simplifies
# down since the dot product of 'cold' indexes in a one-hot encoded vector
# will result in zero. The only term that doesn't become zero
# is the one corresponding to the correct class. So, the loss is simply the
# negative logarithm of the predicted probability of the correct class,
# such that:
#
#                   L = - log(yhat_{i,k})
# Where:
#       L is the loss value
#       yhat_{ik} is the predicted probability for class i, the target label
#       index at k - the index of the correct class.
#
# Note: When referring to log, we are referring to a natural log of base e,
# as in euler's number. A natural logarithm is the inverse of the natural
# exponential function, y = e^x. This means that they 'undo' each other. If
# you take a number x, raise e to that power (e^x), and then take a natural
# log of the result, you get x back. A
# natural logarithm can be written as:
#
#                   y = log_b . x = log_e . x = ln(x)
# Where:
#       y is the exponent or power.
#       b is the base of the logarithm. It's the number that we are raising
#       to a power. The base is always a positive number other than 1. Here,
#       it will be euler's number.
#       x is the argument or number. It's the result you get after raising
#       the base to the power of y.
#
# For example, when trying to find x in the exponential equation for the y
# value of 5.2, such that y = 5.2 = e^x, solving for x, we use the logarithm
# function y = log_e . x = ln(x) where x (input) will be the output of the
# exponential function y such that ln(5.2) = x = 1.64865... You can check the
# answer by inputting it back into the exponential equation y = e^x or y =
# e^{ln(x)}.
#
# How does this fit into a neural network? The final layer of a multi-class
# classification uses softmax to produce a probability distribution. After
# the model makes the prediction, the categorical cross-entropy loss function
# compares this probability distribution to the true one-hot encoded label.
# If the model is confident and correct (e.g., it predicts [0.9, 0.05,
# 0.05] for a class whose true label is [1, 0, 0]), the loss value will be
# very small. If the model is confident but wrong (e.g., it predicts [0.05,
# 0.9, 0.05] for a class whose true label is [1, 0, 0]), the los value will
# be very large. The logarithmic nature of the function heavily penalizes
# confident mistakes.
#
# The loss value is a single number that tells the model how well its
# performing. The goal of training is to minimize this loss. The network uses
# an optimization algorithm to calculate the gradients of the lass with
# respect to the network's weights. These gradients are then used in the
# backpropagation process to update the weights, pushing the model to make
# better predictions and, consequently, lower the loss on the next training
# iteration (more on all of this in p6).

class LayerDense:
    """
    Create a layer of n neurons based on a number of k inputs
    """
    def __init__(self, n_inputs: int, n_neurons: int) -> None:
        self.output = None
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs: np.ndarray | list[list[int | float]]) -> None:
        self.output = np.dot(inputs, self.weights) + self.biases


class ActivationReLU:
    """
    Hidden Layer Activation Function
    """
    def __init__(self) -> None:
        self.output = None

    def forward(self, inputs: np.ndarray | list[list[int | float]]) -> None:
        self.output = np.maximum(0, inputs)


class ActivationSoftmax:
    """
    Output Layer Activation Function
    """
    def __init__(self) -> None:
        self.output = None

    def forward(self, inputs: np.ndarray | list[list[int | float]]) -> None:
        numerator = np.exp((inputs - np.max(inputs, axis=1, keepdims=True)))
        denominator = np.sum(inputs, axis=1, keepdims=True)
        self.output = numerator / denominator


class BrokenLossFunction:
    def __init__(self) -> None:
        self.output = None

    # Notice the issue with this function. When applying log to an output of 0
    # where the class label is true, we get a result of inf, meaning infinity.
    # Mathematically, it is true that the further you get from the correct
    # answer the larger your loss value is and a 100% incorrect confidence or
    # 0% probability is infinitely wrong resulting in an infinitely large
    # loss value. However, when we approach the next step, mean, due to the
    # sensitivity of the mean function to outliers, a single inf output will
    # pull the mean across the vector to infinity.
    def forward(self, class_targets: list[float],
                softmax_outputs: np.ndarray) -> None:
        neg_log = -np.log(softmax_outputs[
                                  range(len(softmax_outputs)), class_targets
                              ])
        self.output = np.mean(neg_log)


try:
    X, y = spiral_data(100, 3)
    h_layer1 = LayerDense(2, 5)
    h_layer2 = LayerDense(5, 5)
    output_layer = LayerDense(5, 3)

    activation1 = ActivationReLU()
    activation2 = ActivationReLU()
    activation3 = ActivationSoftmax()
    loss = LossFunction()

    h_layer1.forward(X)
    activation1.forward(h_layer1.output)
    h_layer2.forward(activation1.output)
    activation2.forward(h_layer2.output)
    output_layer.forward(activation2.output)
    activation3.forward(output_layer.output)
    print(loss.forward(activation3.output))
except Exception as e:
    print(e)