"""
A Neural Net is just a collection of layers.
It behaves like a layer itself, although it will not be one.
"""
from typing import Sequence, Iterator, Tuple
from VisathNN.tensor import Tensor
from VisathNN.layers import Layer

class NeuralNetwork:

    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def backward(self, grad: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    '''
    This is actually a generator, not a Iterator
    '''
    def params_and_grads(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            for name, param in layer.params.items():
                grad = layer.grads[name]
                yield param, grad

