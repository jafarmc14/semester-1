import numpy as np


def generate_weights(neuron, number):
    neuron['weights'] = [np.random.randn() for i in range(number + neuron['bias'])]
    return neuron


def neuron_activation_potential(neuron, inputs):
    activation = 0
    if neuron["bias"]:
        inputs = np.append(inputs, 1)
    for i, weight in enumerate(neuron["weights"]):
        activation += weight * inputs[i]
    return activation


def neuron_linear(neuron, derivative=False):
    out = 0
    if not derivative:
        out = neuron['activation_potential']
    else:
        out = 1
    return out


def neuron_tanh(neuron, derivative=False):
    out = 0
    if not derivative:
        out = (np.exp(neuron['activation_potential']) - np.exp(-neuron['activation_potential'])) / (
                np.exp(neuron['activation_potential']) + np.exp(-neuron['activation_potential']))
    else:
        out = 1.0 - np.power(neuron['output'], 2)
    return out


def neuron_relu(neuron, derivative=False):
    out = 0
    if not derivative:
        out = np.maximum(0, neuron['activation_potential'])
    else:
        if neuron['activation_potential'] >= 0:
            out = 1
    return out


# Sum after all output neurons
def loss_fcn(loss, expected, outputs, derivative=False):
    loss = str.lower(loss)
    error_sum = 0
    if loss == 'mse':
        error_sum = mse(expected,
                        outputs,
                        derivative)
    elif loss == "binary_cross_entropy":
        error_sum = binary_cross_entropy(expected,
                                         outputs,
                                         derivative)
    return error_sum


# Mean Square Error loss function
def mse(expected, outputs, derivative=False):
    error_value = 0
    if not derivative:
        error_value = (expected - outputs) ** 2
    else:
        error_value = expected - outputs
    return error_value


# Cross-entropy loss function
def binary_cross_entropy(expected, outputs, derivative=False):
    error_value = 0
    if not derivative:
        error_value = -expected * np.log(outputs) + (1 - expected) * np.log(outputs)
    else:
        error_value = expected - outputs
    return error_value


n = 5

input = np.random.rand(n, 1)

neuron = {"weights": None,
          "bias": True,
          "activation_potential": None,
          "activation_function": "sigmoid",
          "output": None
          }
print(neuron)

generate_weights(neuron, n)
print(f"Neuron = {neuron}")

t = np.arange(-10, 10, 1)
yout = np.zeros((np.size(t)))

L = []
for (x, y) in zip(t, yout):
    E = loss_fcn("MSE", x, y)
    L.append(E)
print(f"L = {L}")
