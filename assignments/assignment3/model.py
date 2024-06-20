import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        self.Layers = []
        self.Layers = [ConvolutionalLayer(in_channels=3, out_channels=3, filter_size=conv1_channels, padding=1),  #padding?
                        ReLULayer(),
                        MaxPoolingLayer(pool_size=4, stride=4),
                        ConvolutionalLayer(in_channels=3, out_channels=3, filter_size=conv2_channels, padding=1),
                        ReLULayer(),
                        MaxPoolingLayer(pool_size=4, stride=4)]
        
        
        modified_pic_size = self.get_out_size_after_filters(32) # get width (height) of picture modified by filter-like layers

        self.Layers.append(Flattener())

        modified_pic_size = modified_pic_size*modified_pic_size*3 # update width (height) of picture ufter Flattener

        self.Layers.append(FullyConnectedLayer(n_input=modified_pic_size, n_output=n_output_classes))

    def get_out_size_after_filters(self, in_size):
        out_size = in_size
        for layer in self.Layers:
            if type(layer) is ConvolutionalLayer or type(layer) is MaxPoolingLayer:
                out_size = layer.get_out_size(out_size)
        return out_size

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        for key, value in self.params().items():
          value.grad *= 0

        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        for layer in self.Layers:
          X = layer.forward(X)

        loss, dprediction = softmax_with_cross_entropy(X, y)

        dout = dprediction
        for layer in reversed(self.Layers):
          dout = layer.backward(dout)  

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
