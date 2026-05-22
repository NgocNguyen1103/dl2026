import random
class Node:
    def __init__(self, node_imputs):
        self.node_imputs = node_imputs
        self.bias = 0
        self.weights = []
        for _ in range(node_imputs):
            self.weights.append(0)

    def init_bias_weights_random(self):
        self.bias = random.random()
        for _ in range(len(self.weights)):
            self.weights[_] = random.random()

    def init_bias_weights_file(self, bias, weights):
        self.bias = bias
        self.weights = weights

    def sigmoid(self, x):
        e = 2.718281828459045
        return 1 / (1 + e ** (-x))

    def feedforward(self, input_values):
        total_output = self.bias
        for _ in range(len(input_values)):
            total_output = total_output + input_values[_] * self.weights[_]
        final_output = self.sigmoid(total_output)
        if final_output >= 0.5:
            final_output = 1
        else: final_output = 0
        return final_output