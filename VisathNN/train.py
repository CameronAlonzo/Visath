from VisathNN.tensor import Tensor
from VisathNN.nn import NeuralNetwork
from VisathNN.loss import Loss, MSE
from VisathNN.optimizer import Optimizer, SGD
from VisathNN.data import DataIterator, BatchIterator

def train(net: NeuralNetwork,
          inputs: Tensor,
          targets: Tensor,
          epochs: int = 5000,
          iterator: DataIterator = BatchIterator(),
          loss: Loss = MSE(), #Total Squared Error
          optimizer: Optimizer = SGD()) -> None:
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in iterator(inputs, targets):
            predicted = net.forward(batch.inputs)
            epoch_loss += loss.loss(predicted, batch.targets)
            grad = loss.grad(predicted, batch.targets)
            net.backward(grad)
            optimizer.step(net)
        print(epoch, epoch_loss)
