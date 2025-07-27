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

try:
    inputs = [1, 2, 3, 2.5]
    weights = [[0.2, 0.8, -0.5, 1.0],
               [0.5, -0.91, 0.26, -0.5],
               [-0.26, -0.27, 0.17, 0.87]]
    biases = [2, 3, 0.5]
    print(calculate_layer(inputs, weights, biases))
except Exception as e:
    print(e)

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

try:
    inputs = [1, 2, 3, 2.5]
    weights = [[0.2, 0.8, -0.5, 1.0],
               [0.5, -0.91, 0.26, -0.5],
               [-0.26, -0.27, 0.17, 0.87]]
    biases = [2, 3, 0.5]
    print(zip_calculate_layer(inputs, weights, biases))
except Exception as e:
    print(e)