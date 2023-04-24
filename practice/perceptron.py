# Perceptron Architecture : Input Layer, Output Layer
import torch


class Perceptron:
    def __init__(self, nodes, weights):
        self.nodes = nodes
        self.weights = weights

    def input_layer(self):
        print('\nCurrent nodes in Input Layer: ')
        x = self.nodes
        print(x)

    def weighted_nodes(self):
        print('\nRespective weights for input nodes: ')
        y = self.weights
        print(y)

    def activation(self):
        p = self.nodes * self.weights
        print('\nOutput layer: ')
        print(p)


torch.manual_seed(1750)
rn = torch.rand(2,2)
rw = torch.rand(2,2)

myperceptron = Perceptron(rn,rw)

myperceptron.input_layer()

myperceptron.weighted_nodes()

myperceptron.activation()
