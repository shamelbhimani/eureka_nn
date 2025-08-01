import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()
#The problem with a Rectified Linear function is that, because of its
# behaviour in clipping all values <= 0 to 0, if we want to see the
# probability of potential outputs, any negative pass will have a 0%
# probability. In this scenario, we lose nuance on the change in outputs
# between layers of neurons. For example, in a 3-layer neural network,
# if the output of the first layer is all-negative, the 2nd layer will clip
# it, upon activation, to 0, and output all 0s. Further more, the bigger
# problem occurs during backpropagation when, if a neuron's weight is updated
# during backpropagation such that its input is always negative, the ReLU
# activation will always be 0. This means that the gradient for that neuron
# will also be 0, and its weights will stop being updated. The neuron,
# therefore, becomes 'dead' and ceases to learn.

# Here, we add an exponential function. The exponentiation here will allow us
# to retain nuance of negative outputs. How does it do this? Well, when we
# exponentiate an output value such that y = e^x where x is the output from
# layer 1, hence input to layer 2, it prevents the negative values from ever
# truly reaching zero or less than zero. Now how does it do this? Using
# exponentiation in the softmax function, the exponentiation transforms the
# raw outputs of a neural networks (called logits) into a probability
# distribution. The exponential function has a unique and powerful property
# that is key to this solution, as it maps any real number to a positive real
# number. If x > 0, then e^x > 1; if x = 0, then e^x = 1; if x < 0, then e^x
# is always between 0 and 1. So, an input of, let's say -5, becomes e^-5
# =approx. 0.0067, while an input of -0.01 becomes e^-0.01 =approx. 0.99. No
# matter how negative the input is, the output will always be positive and
# non-zero, eliminating the clipping problem and loss of nuance on negative
# values. Once we have the exponentiated output of each individual neuron,
# we will divide these by the summation of all output neurons (called
# normalization) in its layer, giving us the probability distribution of each
# neuron output.

# One thing to mention, and is a problem with the exponentiation function
# itself is that exponentiation of values can easily result in an explosion
# of value sizes as the input sizes increases. For example, an exponentation
# of 1 = euler's number, i.e., 2.71828182..., an exponentation of 100
# =aprox. 2.68811714...e+43 (very large number), and an exponentiation of
# 1000 results in an 'overflow error'. An overflow error is the overflow of
# 'bits', that cannot be processed in memory (most computations occur at a
# maximum of 64 bits). For a 64-bit floating-point number, the largest value
# it can represent is roughly 1.8 x 10^308. Any calculation that results in a
# larger number causes overflow. Here, we need an overflow prevention
# strategy. One clever overflow protection involves subtracting an input
# values in a vector by a constant C PRIOR to exponentiation. Choosing an
# appropriate C is important as it can effect the outputs of each individual
# neuron, its layer, and the whole neural network. For oue purposes, a common
# and easy strategy is to subtract all values in the vector by the maximum
# value in that vector. This results in a vector with a max value of 0 and
# all other values < 0. As you may have guessed, any values <= 0 will be 0 <
# exp^x <= 1, protecting us from an overflow error.
class LayerDense:
    def __init__(self, n_inputs: int, n_neurons: int) -> None:
        self.output = None
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))


    def forward(self, inputs: np.ndarray | list[list[int | float]]) -> None:
        self.output = np.dot(inputs, self.weights) + self.biases


class ActivationReLU:
    def __init__(self):
        self.output = None

    def forward(self, inputs: np.ndarray | list[list[int | float]]) -> None:
        self.output = np.maximum(0, inputs)

class ActivationSoftmax:
    def __init__(self):
        self.output = None

    def forward(self, inputs: np.ndarray | list[list[int | float]]) -> None:
        # for the denominator, ensure in the np.sum() function to set axis=1
        # to sum values in a row (different from pandas, as in a 2d array,
        # axis=1 refers to rows) i.e., all exponentiated values for each
        # individual sample/input value, and keepdims=True to retain the
        # original number of dimensions. A simplified analogy: Say,
        # for example, you have a small batch of 5 students (rows) and their
        # scores on 6 different quizzes (columns), stored in a 2d array of
        # dimensions (5, 6). Now, let's say you want to calculate the average
        # quiz score for each student; you would use np.sum(scores, axis=1)
        # to sum the score for each student (row), and then divide by the
        # number of quizzes (6). When you execute this function without
        # keepdims=True, nmpy collapses or squeezes the dimensions you are
        # summing over into a flat list (5,). It has lost the 'dimension'
        # that represented the columns. Now, if you try to divide the
        # original scores matrix by this vector, numpy will throw an error
        # because the shapes are incompatible (5, 6) cannot be divided by (5,
        # ) properly. When you use keepdims=True, the function keeps the
        # column dimension, resulting in a 2d array (as opposed to a 1d array
        # or flat list) of shape (5, 1) where 1=average of 6 quizzes (
        # columns). So, now when you divide the matrix (5, 6) by the (5,
        # 1) matrix, numpy can easily figure out through broadcasting, i.e.,
        # 'stretching' the (5,1) array to (5,6) (keeping all values as the
        # average of all columns in each column cell). So, for example we
        # have  a couple of rows within the array such that:
        # [  [1, 2, 3, 4, 5, 6],
        #    [7, 8, 9, 10, 11, 12]
        # ]
        # of shape (2, 6) that sum down to an array of (2,1) such that:
        # [  [2.667],
        #    [9.500]
        # ], this output array will stretch back to 6 columns for the
        # element-by-element division process such that:
        # [  [2.667, 2.667, 2.667, 2.667, 2.667, 2.667],
        #    [9.500, 9.500, 9.500, 9.500, 9.500, 9.500]
        # ].
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        denominator = np.sum(exp_values, axis=1, keepdims=True)
        self.output = exp_values / denominator


try:
    X, y = spiral_data(100, 3)

    dense1 = LayerDense(2, 3)
    Activation1=ActivationReLU()

    dense2 = LayerDense(3, 3)
    Activation2=ActivationSoftmax()

    dense1.forward(X)
    Activation1.forward(dense1.output)

    dense2.forward(Activation1.output)
    Activation2.forward(dense2.output)
    print(Activation2.output[:5])
except Exception as e:
    print(e)