"""
The neural net will be made up of layers,
each layer will need a forward and back method
it will pass inputs forward and back propagate gradients
"""
from typing import Dict, Callable
import numpy as np

from DeepLearningLib.tensor import Tensor

class Layer:
    "Default constructor"
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Produce output for inputs
        """
        raise NotImplementedError
    
    def backward(self, grad: Tensor) -> Tensor:
        """
        Propogate gradients backward
        """
        raise NotImplementedError

class Linear(Layer):
    """
    Linear layer computes
    output = input @ weight + bias
    """
    def __init__(self, input_size: int, output_size: int) -> None:
        super()
        #in: batch size, input size
        #out: batch size, output size
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]
    
    def backward(self, grad: Tensor) -> Tensor:
        """
        fundamental calculus
        if y = f(x) and x = a * b + c
        then dy/da = f'(x) * b
        dy/db = f'(x) * a
        dy.dc = f'(x)
        linear algebra
        if y = f(x) and x = a @ b + c
        dy/da = f'(x) @ b.T
        dy/db = a.T @ f'(x)
        dy/dc = f'(x)
        """
        #sum along batch dimension
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T
# F
F = Callable[[Tensor], Tensor]
class Activation(Layer):
    """
    an activation layer applies 
    a function elementwise to the inputs
    """
    #Hyperbolic tanh
    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f # f will take a tensor, and return tensor
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)
    
    def backward(self, grad: Tensor) -> Tensor:
        """
        chain rule element wise
        if y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z)
        """
        return self.f_prime(self.inputs) * grad



def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    """
    The actual math on why this works:
    tanh(x) = e^x - e^-x
              ----------
              e^x + e^-x
              
    d/dx tahnh(x) = 1 - tanh^2(x)
    therefore, if y = tanh(x) then y' with respect to x is
    1 - y^2
    """
    y = tanh(x)
    return 1 - y ** 2
    
class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)
