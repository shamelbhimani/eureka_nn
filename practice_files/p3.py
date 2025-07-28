import numpy as np

# Batch or Parallel Processing allows us to run concurrent processes or
# calculations across processing units in our hardware.

# At the moment, our input is a single sample/row of data and each data point
# in this sample is the feature/column-row-cell of said sample/row. What we
# want to do with batch processing is to pass a batch of samples as opposed
# to one sample at a time. Increasing the batch size helps us fit/generalize
# the data faster. However, it is important to note that showing all samples
# at once may hurt the generalization by overfitting the data. A recommended
# batch size of 32 or 64 is most common.

# Significant note: Remember that in matrix dot product, we multiply the
# points of length l in a row vector r of a matrix p by the points on length k
# in a column vector c in a matrix q. In order for this dot product operation
# to be successful, we need to ensure that the number of features l match
# the number of features k. This can also be interpreted as number of columns
# c_n must be equal to number of rows r_n. So, the index 1 of the shape of the first
# matrix must match the index 0 of the second matrix. matrix_1.shape[1] ==
# matrix_2.shape[0]

def batch_process_error(i: list[list[int | float]],
                  w: list[list[int | float]],
                  b: list[int | float]) -> list[float]:
    """
    The purpose of this function is to throw an error when attempting to
    process or find the dot product of weights and inputs.
    :param i:
    :param w:
    :param b:
    :return:
    """
    output_layer = np.dot(w, i) + b
    return output_layer

#Checks for shape error and transposes inputs so that length of columns of
# weights matches length of rows of inputs, and we are multiplying the right
# features from either matrix.

# The reason why we're transposing weights, and switching inputs and weights
# in np.dot() is because of the way our data is received. We want to ensure
# that when we are checking for shape we are checking that the number of
# features across matrices is the same. Let's assume that inputs has shape (
# samples_n, features_l) and weights has a shape (neurons_n, features_k).
# What we want to do is to multiply feature with feature. Here, we are
# multiplying feature_l with neuron_n, which is not what we want. So,
# we transpose weights so its shape changes to (features_k, neurons_n). Here,
# we will end up multiplying features_l (column) with features_k (rows).
# First, we check if features i.e., column lengths for both i and w do not
# match. This scenario requires us to raise a value error. If they do match (
# there are equal numbers of features across matrices), we will transpose and
# perform the operation.
def batch_process(i: list[list[int | float]],
                  w: list[list[int | float]],
                  b: list[int | float]) -> list[float]:
    if np.array(w).shape[1] != np.array(i).shape[1]:
        raise ValueError(
            f"Input features ({np.array(i).shape[1]}) must match the number "
            f"of input connection for weights ({np.array(w).shape[1]}) for a "
            f"standard layer operations (inputs @ weights.T)"
        )
    else:
        output_layer = np.dot(i, np.array(w).T) + b
    return output_layer

# Shape error (3,4) (3,4) for understanding. You may notice that, at the
# moment, each value at index 0 of each list (column traversal) is attempting
# to multiply with each value in the list at index 0 of list of inputs (row
# traversal).
try:
    inputs = [
        [1, 2, 3, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8],
    ]
    weights = [
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]
    ]
    biases = [2, 3, 0.5]
    batch_process_error(inputs, weights, biases)
except Exception as e:
    print(e, ' ShapeError')

try:
    inputs = [
        [1, 2, 3, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8],
    ]
    weights = [
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]
    ]
    biases = [2, 3, 0.5]
    print(batch_process(inputs, weights, biases))
except Exception as e:
    print(e)
