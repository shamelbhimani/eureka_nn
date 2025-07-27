# Every Neuron in a Neural Network has a unique connection to each previous
# neuron. Let's say there are 3 input neurons, each outputting a value. That
# means, this neuron receives n number of inputs from n number of neurons (
# here, n = 3).

inputs = [1.2, 5.1, 2.1]

# Each input also has a unique weight associated with it. So n number of
# neurons have k number of weights attached, where the cardinality of n and
# the cardinality of k are equal.
weights = [3.1, 2.1, 8.7]

#Additionally, each unique neuron n_i has its own bias attached to it.
bias = 3

#The output for this neuron will be the summation of the dot product of the
# input value at n_i multiplied by the weights value at k_i and you add a
# bias to it.

def calculate(i: list[int | float], w: list[int | float],
              b: int | float) -> int | float:
    total = 0
    for k in range(len(i)):
        total += i[k] * w[k]

    total += b
    return total

# ----------------------------


try:
    print(calculate(inputs, weights, bias))
except:
    print('Something went wrong')
