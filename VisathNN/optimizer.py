'''
We use an optimizer to adust the parameters of our NN
based on the gradients computer during BP
'''
from VisathNN.nn import NeuralNetwork
class Optimizer:
    def step(self, net: NeuralNetwork) -> None:
        raise NotImplementedError
    '''
    lr = Learning Rate -> Lower numbers = better, but more computation required.
    '''
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr
    '''
    function of some tensor, gradient will give direction in which the function increases fastest.
    Reversing this will find decreasing the fastest
    output-loss function will be smaller, hopefully
    '''
    def step(self, net: NeuralNetwork) -> None:
        for param, grad in net.params_and_grads():
            params -= self.lr * grad

    
    