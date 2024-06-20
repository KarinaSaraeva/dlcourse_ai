import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    raise Exception("Not implemented!")

    return loss, grad

def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    if len(predictions.shape) == 1:
      normalised_predictions=predictions - np.amax(predictions)
      probs = np.exp(normalised_predictions)/np.sum(np.exp(normalised_predictions))
    if len(predictions.shape) == 2:
      normalised_predictions=predictions - np.amax(predictions, axis = 1)[:, np.newaxis]
      probs = np.exp(normalised_predictions)/(np.sum(np.exp(normalised_predictions), axis = 1)[:, np.newaxis])
    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    if len(probs.shape) == 1:
      loss = -np.log(probs[target_index])
    if len(probs.shape) == 2:
      target_index = target_index.reshape(target_index.shape[0])
      loss = np.mean(-np.log(np.choose(target_index, probs.T)))
    return  loss

def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO copy from the previous assignment
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    dprediction = np.copy(probs)
    if len(probs.shape) == 1:
      dprediction[target_index] -= 1
    if len(probs.shape) == 2:
      target_index = target_index.reshape(target_index.shape[0])
      dprediction[np.arange(len(target_index)), target_index] -= 1
      dprediction /= len(target_index)
    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.X = X
        result = np.zeros(X.shape)
        result[X > 0] = X[X > 0]
        return result

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_out_d_x = np.zeros(self.X.shape)
        d_out_d_x[self.X > 0] = 1
        d_result = d_out*d_out_d_x
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X
        result = np.dot(X, self.W.value) + self.B.value
        return result

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        self.W.grad = np.dot(self.X.T, d_out)
        self.B.grad = np.dot(np.ones((1, self.X.shape[0])), d_out)
        d_input = np.dot(d_out, self.W.value.T)
        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}


    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def get_out_size(self, in_size):
        return in_size + self.padding*2 - self.filter_size + 1

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X
        padded_X = np.pad(self.X, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant', constant_values=0)

        out_height = self.get_out_size(height)
        out_width = self.get_out_size(width)
        
        result = np.zeros((batch_size, out_height, out_width, self.out_channels))
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        # local_layer  = FullyConnectedLayer(height*width*channels, out_height*out_width*self.out_channels)
        local_W = self.W.value.reshape(-1, self.out_channels)

        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                local_X = padded_X[:, y : (y + self.filter_size), x :(x + self.filter_size), :].reshape(batch_size, -1)
                result[:, y, x, :] = np.dot(local_X, local_W) + self.B.value
        return result


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too
        padded_X = np.pad(self.X, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant', constant_values=0)
        height = height + self.padding*2
        width = width + self.padding*2
        d_input = np.zeros(padded_X.shape)
        self.W.grad = 0

        local_W = self.W.value.reshape(-1, self.out_channels)
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                local_X = padded_X[:, y : (y + self.filter_size), x :(x + self.filter_size), :].reshape(batch_size, -1)
                local_D = d_out[:, y, x, :].reshape(batch_size, out_channels)
                self.W.grad += np.dot(local_X.T, local_D).reshape(self.W.value.shape)
                d_input[:, y : (y + self.filter_size), x : (x + self.filter_size), :] += np.dot(local_D, local_W.T).reshape(batch_size, self.filter_size, self.filter_size, channels)
  
        d_input = d_input[:, self.padding : height-self.padding, self.padding : width-self.padding, :]
        self.B.grad = np.sum(d_out.reshape(-1, out_channels), axis=0)
        return d_input


    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None
        self.mask = None

    def get_out_size(self, in_size):
        return (in_size - self.pool_size) // self.stride + 1

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        self.X = X
        self.mask = np.zeros(X.shape)
        out_height = self.get_out_size(height)
        out_width = self.get_out_size(width)
        result = np.zeros((batch_size, out_height, out_width, channels))
        
        for y in range(out_height):
            for x in range(out_width):
                ystart, xstart = y * self.stride, x * self.stride
                yend, xend = ystart + self.pool_size, xstart + self.pool_size
                result[:, y, x, :] = np.amax(X[:, ystart : yend, xstart : xend, :], axis=(1, 2))
                self.mask[:, ystart : yend, xstart : xend, :] = np.transpose((np.transpose(X[:, ystart : yend, xstart : xend, :], (1, 2, 0, 3)) == result[:, y, x, :]), (2, 0, 1, 3))  
        return result

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, channels = d_out.shape
        dinput = np.zeros(self.X.shape)
       
        for y in range(out_height):
            for x in range(out_width): 
                ystart, xstart = y * self.stride, x * self.stride
                yend, xend = ystart + self.pool_size, xstart + self.pool_size
                local_mask = self.mask[:, ystart : yend, xstart : xend, :]
                din_local = np.zeros(dinput[:, ystart : yend, xstart : xend, :].shape)
                din_local += d_out[:, y:y+1, x:x+1, :]
                din_local *= local_mask
                dinput[:, ystart : yend, xstart : xend, :] += din_local

        return dinput 

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X
        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        result = X.reshape(batch_size, height*width*channels)
        return result

    def backward(self, d_out):
        # TODO: Implement backward pass
        dinput = d_out.reshape(self.X.shape)
        return dinput

    def params(self):
        # No params!
        return {}
