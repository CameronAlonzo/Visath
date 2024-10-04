"""
A loss function simply measures the quality of predictions, we will use this to adjust 
parameters of our model
"""
import numpy as np
from DeepLearningLib.tensor import Tensor

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError
    "Gradient Loss function"
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError
    
class TSE(Loss):
    """
    TSE is total squared error, it is a standard loss function
    This is deprecated, use the MSE function instead for better results
    """
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2)
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)

class MSE(Loss):
    """
    This is the MSE or mean squared error function, it is another standard loss function
    This is preferable over TSE for various reasons
    """
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2)
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual) / predicted.size
    
