import numpy as np

def calculate_neuron(i: list[int | float], weight: list[int | float],
              bias: int | float) -> float:
    total = 0
    for k in range(len(i)):
        total += (float(i[k]) * float(weight[k]))
    total += float(bias)
    return round(total, 2)


def calculate_layer(i: list[int | float],
                    w: list[list[int | float]],
                    b: list[int | float]) -> list[float]:
    output_layer = []
    for j in range(len(w)):
        output_layer.append(calculate_neuron(i, w[j], b[j]))
    return output_layer

# We can also use the zip method in our function.

def zip_calculate_layer(i: list[int | float], w: list[list[int | float]], b: list[int | float]) -> list[float]:
    output_layer = []
    for neuron_weights, neuron_bias in zip(w, b):
        output_layer.append(zip_calculate(i, neuron_weights, neuron_bias))
    return output_layer

def zip_calculate(input_layer: list[int | float],
                  neuron_weight: list[int | float],
                  neuron_bias: int | float) -> float:
    total = 0
    for n_input, weight in zip(input_layer, neuron_weight):
        total += n_input * weight
    total += neuron_bias
    return round(total, 2)

# We can make this process even simpler, and save our effort by using numpy's
# dot product function, 'np.dot(x, y)'. The dot product function allows us to
# sum the products of two vectors at a given location i such that i = 1, 2,
# ..., n. For example the dot product of a vector a = [1, 2] and a vector b =
# [3, 4] is (a_0 * b_0) + (a_1 * b_1).

# A matrix is simply an array of two or more vectors. The way the np.dot
# function works in terms of multiplying matrices to vectors is that the
# first set of values passed into the function must be a matrix. The reason
# for this is that the function is designed to unpack matrices into arrays,
# and then index those arrays when performing the operation so that it goes
# through all the indexes/weights provided in the matrix without throwing a
# shape error. Note, however, the shape of the array in the matrix must match
# the shape of the array passed as the 2nd parameter.


# Notice how the length of the matrix, i.e., the number of weights equals the
# number of biases. This is because the function indexes through the matrix,
# and matches the index of the bias upon calculation.
def dot_calculate_layer(input_layer: list[int | float],
                        matrix_of_weights: list[list[int | float]],
                        list_of_biases: list[int | float]) -> list[float]:
    output_layer = np.dot(matrix_of_weights, input_layer) + list_of_biases
    return output_layer # Notice how we've eliminated a whole calculation
    # function, and reduced the function to an effective 1 line.

try:
    inputs = [1, 2, 3, 2.5]
    weights = [[0.2, 0.8, -0.5, 1.0],
               [0.5, -0.91, 0.26, -0.5],
               [-0.26, -0.27, 0.17, 0.87]]
    biases = [2, 3, 0.5]
    print(calculate_layer(inputs, weights, biases))
    print(zip_calculate_layer(inputs, weights, biases))
    print(dot_calculate_layer(inputs, weights, biases))
except Exception as e:
    print(e)