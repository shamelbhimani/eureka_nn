# Multi-layer processing.
import numpy as np
from p3 import inputs, weights, biases, batch_process

def batch_process_arrays(i: np.ndarray,
                         w: np.ndarray,
                         b: np.ndarray) -> np.ndarray:
    if w.shape[1] != i.shape[1]:
        raise ValueError('Input and output dimensions do not match')
    else:
        output_layer = np.dot(i, w.T) + b

    return output_layer

try:
    inputs_layer_1 = inputs.copy()
    weights_layer_1 = weights.copy()
    biases_layer_1 = biases.copy()

    inputs_layer_2 = batch_process(inputs_layer_1,
                                   weights_layer_1,
                                   biases_layer_1)
    weights_layer_2 = [
        [0.1, -0.14, 0.5],
        [-0.5, 0.12, -0.33],
        [-0.44, 0.73, -0.13]
    ]
    biases_layer_2 = [-1, 2, -0.5]
    print(batch_process_arrays(np.array(inputs_layer_2),
                         np.array(weights_layer_2),
                         np.array(biases_layer_2)))
except Exception as e:
    print(e)
