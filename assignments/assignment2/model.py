import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.Layers = []
        self.Layers = [FullyConnectedLayer(n_input=n_input, n_output=hidden_layer_size), 
                        ReLULayer(), 
                        FullyConnectedLayer(n_input=hidden_layer_size, n_output=n_output)]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!

        for key, value in self.params().items():
          value.grad *= 0
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model

        for layer in self.Layers:
          X = layer.forward(X)

        loss, dprediction = softmax_with_cross_entropy(X, y)

        dout = dprediction
        for layer in reversed(self.Layers):
          dout = layer.backward(dout)  
  
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        for key, value in self.params().items():
          l2_loss, l2_grad = l2_regularization(value.value, self.reg)
          value.grad += l2_grad
          loss += l2_loss

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)

        for layer in self.Layers:
          X = layer.forward(X)
        pred = np.argmax(X, axis=1)
        return pred

    def params(self):
        result = {}
        # TODO Implement aggregating all of the params
        for i in range(len(self.Layers)):
          for key, value in self.Layers[i].params().items():
            result[key+str(i)] = value
        return result
