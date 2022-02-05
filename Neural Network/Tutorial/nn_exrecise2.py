import numpy as np
import matplotlib.pyplot as plt
import csv


class neuron_fcn(object):
    def output(self, neuron, derivative=False):
        """Dispatch method"""
        # Get the method from 'self'. Default to a lambda.
        name = neuron['activation_function']
        method = getattr(self, str(name), "Invalid_function")
        # Call the method as we return it
        return method(neuron, derivative)

    def Invalid_function(self, *arg):
        print("Error: Invalid activation function")
        return None

    def generate_weights(self, neuron, number):
        neuron['weights'] = [np.random.randn() for i in range(number + neuron['bias'])]
        return neuron

    def activation_potential(self, neuron, inputs):
        activation = 0
        if neuron["bias"]:
            inputs = np.append(inputs, 1)
        for i, weight in enumerate(neuron["weights"]):
            activation += weight * inputs[i]
        return activation

    def linear(self, neuron, derivative=False):
        out = 0
        if not derivative:
            out = neuron['activation_potential']
        else:
            out = 1
        return out

    def logistic(self, neuron, derivative=False):
        out = 0
        if not derivative:
            out = 1.0 / (1.0 + np.exp(-neuron['activation_potential']))
        else:
            out = neuron['output'] * (1.0 - neuron['output'])
        return out

    def tanh(self, neuron, derivative=False):
        out = 0
        if not derivative:
            out = (np.exp(neuron['activation_potential']) - np.exp(-neuron['activation_potential'])) / (
                    np.exp(neuron['activation_potential']) + np.exp(-neuron['activation_potential']))
        else:
            out = 1.0 - np.power(neuron['output'], 2)
        return out

    def relu(self, neuron, derivative=False):
        out = 0
        if not derivative:
            out = np.maximum(0, neuron['activation_potential'])
        else:
            if neuron['activation_potential'] >= 0:
                out = 1
        return out


class loss_fcn(object):
    # Error value of neuron
    def loss(self, loss, expected, outputs, derivative):
        """Dispatch method"""
        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, str(loss), lambda: "Invalid_function")
        # Call the method as we return it
        return method(expected, outputs, derivative)

    def Invalid_function(self, *arg):
        print("Error: Invalid loss function")
        return None

    # Mean Square Error loss function
    def mse(self, expected, outputs, derivative=False):
        error_value = 0
        if not derivative:
            error_value = 0.5 * (expected - outputs) ** 2
        else:
            error_value = -(expected - outputs)
        return error_value

    # Cross-entropy loss function
    def binary_cross_entropy(self, expected, outputs, derivative=False):
        error_value = 0
        if not derivative:
            error_value = -expected * np.log(outputs) - (1 - expected) * np.log(1 - outputs)
        else:
            error_value = -(expected / outputs - (1 - expected) / (1 - outputs))
        # print(f"output = {outputs}, expected = {expected}, error = {error_value}, derivative = {derivative}")
        return error_value


# Initialize a network
class neuron_network(object):
    # Forward propagate input to a network output
    def forward_propagate(self, neuron, inputs):
        bias_inputs = inputs.copy()
        if neuron['bias']:
            np.append(bias_inputs, 1)
        tf = neuron_fcn()
        neuron['activation_potential'] = tf.activation_potential(neuron, inputs)
        neuron['output'] = tf.output(neuron, derivative=False)
        return neuron['output']

    # Backpropagate error and store it in neuron
    def backward_propagate(self, loss_function, neuron, expected):
        error = loss_fcn().loss(loss_function, expected, neuron['output'], derivative=True)
        n = neuron_fcn()
        neuron['delta'] = error * n.output(neuron, derivative=True)

    # Update network weights with error
    def update_weights(self, neuron, inputs, l_rate):
        if len(np.shape(inputs)):
            for j in range(len(inputs)):
                neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j]
            if neuron['bias']:
                neuron['weights'][-1] -= l_rate * neuron['delta']
        else:
            neuron['weights'][0] -= l_rate * neuron['delta'] * inputs
            if neuron['bias']:
                neuron['weights'][1] -= l_rate * neuron['delta']

    # Train a network for a fixed number of epochs
    def train(self, neuron, x_train, y_train, l_rate=0.01, n_epoch=100, loss_function='mse', verbose=1):
        for epoch in range(n_epoch):
            sum_error = 0
            for iter, (x_row, y_row) in enumerate(zip(x_train, y_train)):
                outputs = self.forward_propagate(neuron, x_row)

                loss = loss_fcn()
                l = loss.loss(loss_function, y_row, outputs, derivative=False)
                sum_error += l
                if verbose > 1:
                    print(f"iteration = {iter+1}, output = {outputs:.4f}, target = {y_row:.4f}, loss = {l:.4f}")

                self.backward_propagate(loss_function, neuron, y_row)

                self.update_weights(neuron, x_row, l_rate)

            if verbose > 0:
                print('>epoch=%d, loss=%.3f' % (epoch+1, sum_error))
        return neuron

    # Calculate network output
    def predict(self, neuron, inputs):
        y = []
        for input in inputs:
            y.append(self.forward_propagate(neuron, input))
        return y


def generate_regression_data(n, tosave=True, fname="reg_data"):
    # Generate regression dataset
    X = np.linspace(0, 10, n)
    Y = 2 * X + 4 * np.random.rand(np.size(X)) - 2

    plt.plot(X, Y, 'r--o', label="Training data")
    plt.legend()
    plt.grid()
    plt.show()

    np.savetxt('X_data.dat', X)
    np.savetxt('Y_data.dat', Y)

    return X.reshape(-1, 1).tolist(), Y.reshape(-1, 1).tolist()


def read_regression_data(fname="reg_data"):
    X = np.loadtxt('X_data.dat')
    Y = np.loadtxt('Y_data.dat')

    plt.plot(X, Y, 'r--o', label="Training data")
    plt.legend()
    plt.grid()
    plt.show()

    return X, Y


def test_regression():
    # Read data
    X, Y = read_regression_data()

    # Building one neuron neural network
    neuron = {"weights": None,
              "bias": True,
              "activation_potential": None,
              "activation_function": "linear",
              'delta': None,
              "output": None
              }
    print(neuron)

    neuron_fcn().generate_weights(neuron, 1)
    print(f"G1 = {neuron}")

    model = neuron_network()
    model.train(neuron, X, Y, 0.001, 1000, 'mse')
    print(f"G1 = {neuron}")

    X_test = np.linspace(0, 10, 100).reshape(-1, 1)
    X_test = np.array(X_test).tolist()

    predicted = model.predict(neuron, X_test)

    plt.plot(X, Y, 'r--o', label="Training data")
    plt.plot(X_test, predicted, 'b--x', label="Predicted")
    plt.legend()
    plt.grid()
    plt.show()


def generate_classification_data(n=30, tosave=True, fname="class_data"):
    # Class 1 - samples generation
    X1_1 = 1 + 4 * np.random.rand(n, 1)
    X1_2 = 1 + 4 * np.random.rand(n, 1)
    class1 = np.concatenate((X1_1, X1_2), axis=1)
    Y1 = np.ones(n)

    # Class 0 - samples generation
    X0_1 = 3 + 4 * np.random.rand(n, 1)
    X0_2 = 3 + 4 * np.random.rand(n, 1)
    class0 = np.concatenate((X0_1, X0_2), axis=1)
    Y0 = np.zeros(n)

    X = np.concatenate((class1, class0))
    Y = np.concatenate((Y1, Y0))

    idx0 = [i for i, v in enumerate(Y) if v == 0]
    idx1 = [i for i, v in enumerate(Y) if v == 1]

    plt.scatter(X[idx1, 0], X[idx1, 1], marker='^', c="red", label="class 1")
    plt.scatter(X[idx0, 0], X[idx0, 1], marker='o', c="blue", label="class 0")
    plt.legend()
    plt.show()

    np.savetxt('X_data.dat', X)
    np.savetxt('Y_data.dat', Y)

    return X, Y, idx0, idx1

def read_classification_data(fname="class_data"):
    X = np.loadtxt('X_data.dat')
    Y = np.loadtxt('Y_data.dat')

    idx0 = [i for i, v in enumerate(Y) if v == 0]
    idx1 = [i for i, v in enumerate(Y) if v == 1]

    plt.scatter(X[idx1, 0], X[idx1, 1], marker='^', c="red", label="class 1")
    plt.scatter(X[idx0, 0], X[idx0, 1], marker='o', c="blue", label="class 0")
    plt.legend()
    plt.show()

    return X, Y, idx0, idx1


def test_classification():
    # Read data
    X, Y, idx0, idx1 = read_classification_data()

    # Building one neuron neural network
    neuron = {"weights": None,
              "bias": True,
              "activation_potential": None,
              "activation_function": "logistic",
              'delta': None,
              "output": None
              }
    print(neuron)

    neuron_fcn().generate_weights(neuron, 2)
    print(f"G1 = {neuron}")

    model = neuron_network()
    model.train(neuron, X, Y, 0.1, 100, 'binary_cross_entropy', 2)
    # print(f"G1 = {neuron}")

    plt.scatter(X[idx1, 0], X[idx1, 1], marker='^', c="red", label="class 1")
    plt.scatter(X[idx0, 0], X[idx0, 1], marker='o', c="blue", label="class 0")

    X_line = np.linspace(0, 10, 100).reshape(-1, 1)
    Y_line = -neuron["weights"][1] / neuron["weights"][0] * X_line - neuron["weights"][2] / neuron["weights"][0]
    plt.plot(X_line, Y_line, label="Class border")
    plt.legend()
    plt.show()

    sum = 0
    y = model.predict(neuron, X)
    T = np.sum(1 - np.abs(np.round(np.array(y)) - np.array(Y)))

    ACC = T/len(X)
    print(f"Classification accuracy = {ACC}")

generate_classification_data()
test_classification()

# generate_regression_data(30)
# test_regression()
