import Node
class Layer:
    def __init__(self, layer_nodes, node_inputs):
        self.layer_nodes = layer_nodes
        self.node_inputs = node_inputs
        self.nodes = []
        for _ in range(layer_nodes):
            node = Node(node_inputs)
            self.nodes.append(node)

    def init_random(self):
        for _ in range(len(self.nodes)):
            self.nodes[_].init_bias_weights_random()

    def feedforward(self, input_values):
        layer_outputs = []
        for _ in range(len(self.nodes)):
            node_output = self.nodes[_].feedforward(input_values)
            layer_outputs.append(node_output)
        return layer_outputs